import json
import logging
import math
import os
import shutil
import re
import struct
import tempfile
import zlib
from fractions import Fraction

import av
import comfy.utils
import folder_paths
import nodes
import numpy as np
import torch
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc
from comfy.cli_args import args
from comfy_api.latest import IO, UI, ComfyExtension
from PIL.Image import Exif

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# ---------------------------------------------------------------------------
# Format specifications
# ---------------------------------------------------------------------------

# Maps (file_format, bit_depth, num_channels) -> (quantization scale, numpy dtype,
# av frame pix_fmt, stream pix_fmt). Keeps the encode path declarative instead of branchy.
_FORMAT_SPECS = {
    ("png", "8-bit", 1): {"scale": 255.0, "dtype": np.uint8, "frame_fmt": "gray", "stream_fmt": "gray"},
    ("png", "8-bit", 3): {"scale": 255.0, "dtype": np.uint8, "frame_fmt": "rgb24", "stream_fmt": "rgb24"},
    ("png", "8-bit", 4): {"scale": 255.0, "dtype": np.uint8, "frame_fmt": "rgba", "stream_fmt": "rgba"},
    ("png", "16-bit", 1): {"scale": 65535.0, "dtype": np.uint16, "frame_fmt": "gray16le", "stream_fmt": "gray16be"},
    ("png", "16-bit", 3): {"scale": 65535.0, "dtype": np.uint16, "frame_fmt": "rgb48le", "stream_fmt": "rgb48be"},
    ("png", "16-bit", 4): {"scale": 65535.0, "dtype": np.uint16, "frame_fmt": "rgba64le", "stream_fmt": "rgba64be"},
    ("exr", "32-bit float", 1): {"scale": 1.0, "dtype": np.float32, "frame_fmt": "grayf32le", "stream_fmt": "grayf32le"},
    ("exr", "32-bit float", 3): {"scale": 1.0, "dtype": np.float32, "frame_fmt": "gbrpf32le", "stream_fmt": "gbrpf32le"},
    ("exr", "32-bit float", 4): {"scale": 1.0, "dtype": np.float32, "frame_fmt": "gbrapf32le", "stream_fmt": "gbrapf32le"},
}

_AVIF_COLOR_PROPERTIES = {
    "sRGB": (ColorPrimaries.BT709, ColorTrc.IEC61966_2_1, 1),
    "HDR": (ColorPrimaries.BT2020, ColorTrc.ARIB_STD_B67, 9),
    "HDR PQ": (ColorPrimaries.BT2020, ColorTrc.SMPTE2084, 9),
}


# ---------------------------------------------------------------------------
# Color transforms
# ---------------------------------------------------------------------------


def srgb_to_linear(t: torch.Tensor) -> torch.Tensor:
    """Inverse sRGB EOTF (IEC 61966-2-1). Operates on RGB channels only;
    alpha (if present as the 4th channel) is passed through unchanged."""
    if t.shape[-1] == 4:
        rgb, alpha = t[..., :3], t[..., 3:]
        return torch.cat([srgb_to_linear(rgb), alpha], dim=-1)

    # Piecewise: linear toe below 0.04045, gamma curve above.
    low = t / 12.92
    high = ((t.clamp(min=0.0) + 0.055) / 1.055) ** 2.4
    return torch.where(t <= 0.04045, low, high)


# HLG OETF constants from BT.2100 Table 5.
_HLG_A = 0.17883277
_HLG_B = 0.28466892
_HLG_C = 0.55991072928  # = 0.5 - a*ln(4*a)


def hlg_to_linear(t: torch.Tensor) -> torch.Tensor:
    """Inverse HLG OETF (BT.2100). Maps a non-linear HLG signal in [0, 1] to
    *scene*-linear light in [0, 1]. Per BT.2100 Note 5a, this is the correct
    transform when converting HLG to a linear scene-light representation
    (rather than display-light, which would also involve the HLG OOTF).

    Operates on RGB channels only; alpha is passed through unchanged."""
    if t.shape[-1] == 4:
        rgb, alpha = t[..., :3], t[..., 3:]
        return torch.cat([hlg_to_linear(rgb), alpha], dim=-1)

    # Piecewise: sqrt branch below 0.5, log branch above.
    # Clamp the log branch at the 0.5 branch point (not above it) so the
    # unselected lane stays finite in exp() without altering selected values;
    # values above 1.0 are allowed and extrapolate naturally.
    low = (t**2) / 3.0
    high = (torch.exp((t.clamp(min=0.5) - _HLG_C) / _HLG_A) + _HLG_B) / 12.0
    return torch.where(t <= 0.5, low, high)


_REC709_TO_REC2020 = (
    (0.6274038959346991, 0.3292830383778837, 0.0433130656874172),
    (0.0690972893582320, 0.9195403950754587, 0.0113623155663092),
    (0.0163914388751503, 0.0880133078772259, 0.8955952532476238),
)
_REC2020_TO_REC709 = (
    (1.6604910021084338, -0.5876411387885494, -0.0728498633198846),
    (-0.1245504745215905, 1.1328998971259600, -0.0083494226043695),
    (-0.0181507633549053, -0.1005788980080076, 1.1187296613629125),
)
_REC709_LUMA = (0.2126390058715103, 0.7151686787677559, 0.0721923153607337)
_REC2020_LUMA = (0.2627, 0.6780, 0.0593)
_PQ_M1, _PQ_M2 = 2610 / 16384, 2523 / 32
_PQ_C1, _PQ_C2, _PQ_C3 = 3424 / 4096, 2413 / 128, 2392 / 128
_SDR_WHITE_NITS = 203.0
_HLG_PEAK_NITS = 1000.0
_HLG_GAMMA = 1.2


def _convert_rgb_primaries(rgb, matrix):
    r, g, b = rgb.unbind(dim=-1)
    return torch.stack([r * row[0] + g * row[1] + b * row[2] for row in matrix], dim=-1)


def _rgb_luminance(rgb, weights):
    return (rgb * rgb.new_tensor(weights)).sum(dim=-1, keepdim=True)


def _tone_map_luminance(rgb, weights):
    luminance = _rgb_luminance(rgb, weights).clamp_min(0.0)
    # Extended Reinhard, sharing a white point across the batch to avoid frame-by-frame exposure changes.
    peak = luminance.amax().clamp_min(1.0)
    scale = (1.0 + luminance / peak.square()) / (1.0 + luminance)
    # Allow transfer-function roundoff at SDR white without engaging tone mapping.
    return rgb * torch.where(peak > 1.0001, scale, 1.0)


def _compress_rgb_gamut(rgb, weights):
    luminance = _rgb_luminance(rgb, weights).clamp(0.0, 1.0)
    chroma = rgb - luminance
    tiny = torch.finfo(rgb.dtype).tiny
    minimum = rgb.amin(dim=-1, keepdim=True)
    maximum = rgb.amax(dim=-1, keepdim=True)
    upper = (1.0 - luminance) / (maximum - luminance).clamp_min(tiny)
    lower = luminance / (luminance - minimum).clamp_min(tiny)
    saturation = torch.minimum(upper, lower).clamp(0.0, 1.0)
    # Do not desaturate boundary colors for transfer-function roundoff.
    in_gamut = (minimum >= -1e-5) & (maximum <= 1.00001)
    return torch.where(in_gamut, rgb, torch.addcmul(luminance, chroma, saturation)).clamp(0.0, 1.0)


class ImageColorSpace(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        spaces = ["sRGB", "HDR", "HDR PQ", "linear"]
        return IO.Schema(
            node_id="ImageColorSpace",
            display_name="Convert Image Color Space",
            category="image/color",
            description="Convert sRGB, linear Rec.709, HDR (Rec.2020 HLG), and HDR PQ (Rec.2020 PQ). Linear 1.0 uses the same 203-nit reference white as sRGB; HLG uses a 1000-nit reference display. Linear output and linear-to-HDR conversions preserve extended values without tone mapping. SDR output and PQ-to-HLG conversion tone-map excess luminance across the batch and compress out-of-gamut colors. Conversions compute in float32 and return the intermediate device and dtype. Straight alpha is not color-transformed.",
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("source", options=spaces, default="sRGB", tooltip="Color space of the input pixels."),
                IO.Combo.Input(
                    "destination",
                    options=spaces,
                    default="sRGB",
                    tooltip="Color space of the output pixels. Set the save node to this same color space.",
                ),
            ],
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, source, destination) -> IO.NodeOutput:
        if source == destination:
            return IO.NodeOutput(
                image.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
            )

        # PQ's exponents and near-cancelling constants need more precision than float16/bfloat16.
        rgb = image[..., :3].float()

        # Convert to display-linear Rec.2020 in cd/m² (BT.2100 EOTFs).
        if source == "sRGB":
            rgb = _convert_rgb_primaries(srgb_to_linear(rgb), _REC709_TO_REC2020) * _SDR_WHITE_NITS
        elif source == "linear":
            rgb = _convert_rgb_primaries(rgb, _REC709_TO_REC2020) * _SDR_WHITE_NITS
        elif source == "HDR":
            rgb = hlg_to_linear(rgb)
            luminance = _rgb_luminance(rgb, _REC2020_LUMA).clamp_min(0.0)
            rgb = rgb * (luminance.pow(_HLG_GAMMA - 1.0) * _HLG_PEAK_NITS)
        elif source == "HDR PQ":
            # Evaluate PQ around 1 to avoid cancellation in float32.
            p = (rgb.clamp_min(0.0).log() / _PQ_M2).expm1()
            rgb = ((p + (1.0 - _PQ_C1)).clamp_min(0.0) / ((_PQ_C2 - _PQ_C3) - _PQ_C3 * p)).pow(1.0 / _PQ_M1) * 10000.0
        else:
            raise ValueError(f"Unsupported source color space: {source}")

        if destination == "linear":
            rgb = _convert_rgb_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_REC709)
        elif destination == "sRGB":
            rgb = _convert_rgb_primaries(rgb / _SDR_WHITE_NITS, _REC2020_TO_REC709)
            rgb = _tone_map_luminance(rgb, _REC709_LUMA)
            rgb = _compress_rgb_gamut(rgb, _REC709_LUMA)
            rgb = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb.pow(1.0 / 2.4) - 0.055)
        elif destination == "HDR":
            rgb = rgb / _HLG_PEAK_NITS
            if source == "HDR PQ":
                rgb = _tone_map_luminance(rgb, _REC2020_LUMA)
            luminance = _rgb_luminance(rgb, _REC2020_LUMA).clamp_min(torch.finfo(rgb.dtype).tiny)
            rgb = rgb * luminance.pow(1.0 / _HLG_GAMMA - 1.0)
            if source == "HDR PQ":
                rgb = _compress_rgb_gamut(rgb, _REC2020_LUMA)
            low = (3.0 * rgb.clamp_min(0.0)).sqrt()
            high = _HLG_A * (12.0 * rgb.clamp_min(1.0 / 12.0) - _HLG_B).log() + _HLG_C
            rgb = torch.where(rgb <= 1.0 / 12.0, low, high)
        elif destination == "HDR PQ":
            p = (rgb.clamp_min(0.0) / 10000.0).pow(_PQ_M1)
            p = ((_PQ_C1 - 1.0) + (_PQ_C2 - _PQ_C3) * p) / (1.0 + _PQ_C3 * p)
            rgb = (p.log1p() * _PQ_M2).exp()
        else:
            raise ValueError(f"Unsupported destination color space: {destination}")

        if image.shape[-1] == 4:
            rgb = torch.cat((rgb, image[..., 3:]), dim=-1)
        return IO.NodeOutput(rgb.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype()))


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a single PNG chunk: length | type | data | CRC32(type+data)."""
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _png_text_chunk(keyword: str, text: str) -> bytes:
    """tEXt chunk: latin-1 keyword + NUL + latin-1 text."""
    payload = keyword.encode("latin-1") + b"\x00" + text.encode("latin-1", errors="replace")
    return _png_chunk(b"tEXt", payload)


def inject_png_metadata(png_bytes: bytes, prompt: dict | None, extra_pnginfo: dict | None) -> bytes:
    """Insert ComfyUI prompt/workflow as tEXt chunks right after IHDR."""
    if not png_bytes.startswith(_PNG_SIGNATURE):
        return png_bytes

    chunks: list[bytes] = []
    if prompt is not None:
        chunks.append(_png_text_chunk("prompt", json.dumps(prompt)))
    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            chunks.append(_png_text_chunk(key, json.dumps(value)))
    if not chunks:
        return png_bytes

    # IHDR is always the first chunk; insert ours immediately after it.
    ihdr_length = struct.unpack(">I", png_bytes[8:12])[0]
    ihdr_end = 8 + 8 + ihdr_length + 4  # signature + (len+type) + data + crc
    return png_bytes[:ihdr_end] + b"".join(chunks) + png_bytes[ihdr_end:]


# Standard chromaticities (CIE 1931 xy) for the colorspaces this node writes.
# Each tuple is (Rx, Ry, Gx, Gy, Bx, By, Wx, Wy). All share D65 white point.
_CHROMATICITIES = {
    # ITU-R BT.709 / sRGB primaries
    "Rec.709": (0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600, 0.3127, 0.3290),
    # ITU-R BT.2020 (UHDTV / wide-gamut HDR) primaries
    "Rec.2020": (0.7080, 0.2920, 0.1700, 0.7970, 0.1310, 0.0460, 0.3127, 0.3290),
}


def _pack_chromaticities(primaries: tuple) -> bytes:
    """Serialize 8 chromaticity floats into the EXR `chromaticities` payload."""
    return struct.pack("<8f", *primaries)


def _exr_attribute(name: str, attr_type: str, value: bytes) -> bytes:
    """Serialize one EXR header attribute: name\\0 type\\0 size:int32 value."""
    return name.encode("utf-8") + b"\x00" + attr_type.encode("utf-8") + b"\x00" + struct.pack("<i", len(value)) + value


def inject_exr_metadata(
    exr_bytes: bytes,
    prompt: dict | None,
    extra_pnginfo: dict | None,
    colorspace: str | None = None,
) -> bytes:
    """Insert ComfyUI metadata and color-space info into an EXR header.

    Color: EXR pixels are linear by convention. The standard way to describe
    their RGB→XYZ relationship is the `chromaticities` attribute. We pick the
    primaries that match what the user told us their input was:

      colorspace="sRGB" → Rec. 709 / sRGB primaries (D65)
      colorspace="HDR"  → Rec. 2020 / BT.2100 primaries (D65)

    Pixels are always converted to linear scene light upstream (sRGB EOTF
    inverse for sRGB; HLG OETF inverse for HDR), so the file content is
    scene-linear in the indicated gamut. OpenEXR has no standard transfer-
    function attribute (the OpenEXR TSC has discussed adding one but it
    doesn't exist), so we don't invent one — `chromaticities` plus the EXR
    linear-by-convention rule fully specifies the color.

    Prompt/workflow: written as plain `string` attributes using the same keys
    (`prompt`, `workflow`, ...) that Comfy uses for PNG tEXt chunks, so the
    same readers can pull them out symmetrically.

    Implementation note: the chunk-offset table that follows the header stores
    *absolute* byte offsets into the file. Inserting N bytes into the header
    means every offset must be incremented by N or the file becomes unreadable.
    """
    if len(exr_bytes) < 8 or exr_bytes[:4] != b"\x76\x2f\x31\x01":
        return exr_bytes

    new_blob = b""
    if prompt is not None:
        new_blob += _exr_attribute("prompt", "string", json.dumps(prompt).encode("utf-8"))
    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            new_blob += _exr_attribute(key, "string", json.dumps(value).encode("utf-8"))
    if colorspace is not None:
        # Map each colorspace option to the RGB primaries the linear pixels
        # are now in. "sRGB" and "linear" both produce Rec. 709 linear; "HDR"
        # (HLG-encoded Rec. 2020 input) produces Rec. 2020 linear.
        primaries_name = {
            "sRGB": "Rec.709",
            "linear": "Rec.709",
            "HDR": "Rec.2020",
        }.get(colorspace, "Rec.709")
        new_blob += _exr_attribute(
            "chromaticities",
            "chromaticities",
            _pack_chromaticities(_CHROMATICITIES[primaries_name]),
        )
    if not new_blob:
        return exr_bytes

    # Walk header attributes to find the terminating null byte, and pick up
    # dataWindow + compression so we know how many chunks the offset table has.
    pos = 8  # past magic (4) + version (4)
    data_window = None
    compression = 0
    while pos < len(exr_bytes) and exr_bytes[pos] != 0:
        name_end = exr_bytes.index(b"\x00", pos)
        attr_name = exr_bytes[pos:name_end].decode("latin-1", errors="replace")
        type_end = exr_bytes.index(b"\x00", name_end + 1)
        attr_type = exr_bytes[name_end + 1 : type_end].decode("latin-1", errors="replace")
        size = struct.unpack("<i", exr_bytes[type_end + 1 : type_end + 5])[0]
        value_start = type_end + 5
        value = exr_bytes[value_start : value_start + size]

        if attr_name == "dataWindow" and attr_type == "box2i":
            data_window = struct.unpack("<iiii", value)  # xMin, yMin, xMax, yMax
        elif attr_name == "compression" and attr_type == "compression":
            compression = value[0]

        pos = value_start + size

    if data_window is None:
        return exr_bytes  # required attribute missing — don't risk corrupting

    # Scanlines per chunk by compression, from the OpenEXR spec.
    scanlines_per_block = {
        0: 1,  # NO_COMPRESSION
        1: 1,  # RLE
        2: 1,  # ZIPS
        3: 16,  # ZIP
        4: 32,  # PIZ
        5: 16,  # PXR24
        6: 32,  # B44
        7: 32,  # B44A
        8: 256,  # DWAA
        9: 256,  # DWAB
    }.get(compression, 1)

    _, y_min, _, y_max = data_window
    height = y_max - y_min + 1
    num_chunks = (height + scanlines_per_block - 1) // scanlines_per_block

    header_end = pos  # position of the terminating null byte
    table_start = header_end + 1
    pixel_start = table_start + num_chunks * 8
    delta = len(new_blob)

    old_offsets = struct.unpack(f"<{num_chunks}Q", exr_bytes[table_start:pixel_start])
    new_table = struct.pack(f"<{num_chunks}Q", *(o + delta for o in old_offsets))

    return (
        exr_bytes[:header_end]  # header attributes
        + new_blob  # our new attributes
        + exr_bytes[header_end:table_start]  # terminating null byte
        + new_table  # shifted offset table
        + exr_bytes[pixel_start:]  # pixel data, untouched
    )


def _bmff_box(box_type: bytes, payload: bytes) -> bytes:
    size = 8 + len(payload)
    if size > 0xFFFFFFFF:
        raise ValueError("AVIF metadata box is too large.")
    return struct.pack(">I4s", size, box_type) + payload


def _bmff_boxes(data: bytes, start: int, end: int) -> list[tuple[int, int, bytes, int]]:
    boxes = []
    pos = start
    while pos < end:
        if pos + 8 > end:
            raise ValueError("Invalid AVIF box structure.")
        size, box_type = struct.unpack_from(">I4s", data, pos)
        header_size = 8
        if size == 1:
            if pos + 16 > end:
                raise ValueError("Invalid AVIF extended-size box.")
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            header_size = 16
        elif size == 0:
            size = end - pos
        if size < header_size or pos + size > end:
            raise ValueError("Invalid AVIF box size.")
        boxes.append((pos, size, box_type, header_size))
        pos += size
    return boxes


def _avif_exif(metadata: dict) -> bytes:
    exif = Exif()
    if "prompt" in metadata:
        exif[0x0110] = f"prompt:{json.dumps(metadata['prompt'])}"
    next_tag = 0x010F
    for key, value in metadata.items():
        if key == "prompt":
            continue
        exif[next_tag] = f"{key}:{json.dumps(value)}"
        next_tag -= 1
    return b"\x00\x00\x00\x00" + exif.tobytes()[6:]


def _add_avif_exif_item(meta: bytes, exif_offset: int, exif_length: int, offset_delta: int) -> bytes:
    if meta[4:8] != b"meta" or len(meta) < 12:
        raise ValueError("AVIF metadata requires a valid meta box.")
    children = _bmff_boxes(meta, 12, len(meta))
    child_by_type = {box_type: (pos, size, header_size) for pos, size, box_type, header_size in children}
    if not all(box_type in child_by_type for box_type in (b"pitm", b"iloc", b"iinf")):
        raise ValueError("AVIF metadata boxes are incomplete.")

    pitm_pos, pitm_size, _ = child_by_type[b"pitm"]
    pitm = meta[pitm_pos : pitm_pos + pitm_size]
    if pitm[8] != 0:
        raise ValueError("Unsupported AVIF primary-item format.")
    primary_item_id = struct.unpack_from(">H", pitm, 12)[0]

    iloc_pos, iloc_size, _ = child_by_type[b"iloc"]
    iloc = bytearray(meta[iloc_pos : iloc_pos + iloc_size])
    if iloc[8] != 0 or iloc[12] != 0x44 or iloc[13] != 0:
        raise ValueError("Unsupported AVIF item-location format.")
    item_count = struct.unpack_from(">H", iloc, 14)[0]
    cursor = 16
    item_ids = []
    for _ in range(item_count):
        item_id, _, extent_count = struct.unpack_from(">HHH", iloc, cursor)
        cursor += 6
        item_ids.append(item_id)
        for _ in range(extent_count):
            extent_offset = struct.unpack_from(">I", iloc, cursor)[0]
            struct.pack_into(">I", iloc, cursor, extent_offset + offset_delta)
            cursor += 8
    if cursor != len(iloc):
        raise ValueError("Unsupported AVIF item-location entries.")
    exif_item_id = max(item_ids) + 1
    if exif_item_id > 0xFFFF or exif_offset > 0xFFFFFFFF or exif_length > 0xFFFFFFFF:
        raise ValueError("AVIF metadata exceeds 32-bit item limits.")
    struct.pack_into(">H", iloc, 14, item_count + 1)
    iloc.extend(struct.pack(">HHHII", exif_item_id, 0, 1, exif_offset, exif_length))
    struct.pack_into(">I", iloc, 0, len(iloc))

    iinf_pos, iinf_size, _ = child_by_type[b"iinf"]
    iinf = bytearray(meta[iinf_pos : iinf_pos + iinf_size])
    if iinf[8] != 0:
        raise ValueError("Unsupported AVIF item-information format.")
    iinf_count = struct.unpack_from(">H", iinf, 12)[0]
    struct.pack_into(">H", iinf, 12, iinf_count + 1)
    infe_payload = b"\x02\x00\x00\x00" + struct.pack(">HH4s", exif_item_id, 0, b"Exif") + b"\x00"
    iinf.extend(_bmff_box(b"infe", infe_payload))
    struct.pack_into(">I", iinf, 0, len(iinf))

    cdsc = _bmff_box(b"cdsc", struct.pack(">HHH", exif_item_id, 1, primary_item_id))
    if b"iref" in child_by_type:
        iref_pos, iref_size, _ = child_by_type[b"iref"]
        iref = bytearray(meta[iref_pos : iref_pos + iref_size])
        if iref[8] != 0:
            raise ValueError("Unsupported AVIF item-reference format.")
        iref.extend(cdsc)
        struct.pack_into(">I", iref, 0, len(iref))
    else:
        iref = _bmff_box(b"iref", b"\x00\x00\x00\x00" + cdsc)

    output = bytearray(meta[:12])
    for pos, size, box_type, _ in children:
        if box_type == b"iloc":
            output.extend(iloc)
        elif box_type == b"iinf":
            output.extend(iinf)
            if b"iref" not in child_by_type:
                output.extend(iref)
        elif box_type == b"iref":
            output.extend(iref)
        else:
            output.extend(meta[pos : pos + size])
    struct.pack_into(">I", output, 0, len(output))
    return bytes(output)


def _adjust_avif_chunk_offsets(moov: bytes, offset_delta: int) -> bytes:
    output = bytearray(moov)
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}

    def adjust(start: int, end: int) -> None:
        for pos, size, box_type, header_size in _bmff_boxes(output, start, end):
            if box_type in containers:
                adjust(pos + header_size, pos + size)
            elif box_type in (b"stco", b"co64"):
                entry_size = 4 if box_type == b"stco" else 8
                entry_count = struct.unpack_from(">I", output, pos + header_size + 4)[0]
                cursor = pos + header_size + 8
                if cursor + entry_count * entry_size != pos + size:
                    raise ValueError("Invalid AVIF chunk-offset table.")
                value_format = ">I" if entry_size == 4 else ">Q"
                for _ in range(entry_count):
                    value = struct.unpack_from(value_format, output, cursor)[0]
                    struct.pack_into(value_format, output, cursor, value + offset_delta)
                    cursor += entry_size

    adjust(0, len(output))
    return bytes(output)


def _copy_file_bytes(source, destination, size: int) -> None:
    remaining = size
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("Unexpected end of AVIF file.")
        destination.write(chunk)
        remaining -= len(chunk)


def inject_avif_metadata(path: str, metadata: dict) -> None:
    """Add a ComfyUI-compatible EXIF item without loading media data into memory."""
    if not metadata:
        return
    exif = _avif_exif(metadata)
    file_size = os.path.getsize(path)
    top_level_boxes = []
    with open(path, "rb") as source:
        pos = 0
        while pos < file_size:
            source.seek(pos)
            header = source.read(8)
            if len(header) != 8:
                raise ValueError("Invalid AVIF box header.")
            size, box_type = struct.unpack(">I4s", header)
            header_size = 8
            size_field = size
            if size == 1:
                extended_size = source.read(8)
                if len(extended_size) != 8:
                    raise ValueError("Invalid AVIF extended-size box.")
                size = struct.unpack(">Q", extended_size)[0]
                header_size = 16
            elif size == 0:
                size = file_size - pos
            if size < header_size or pos + size > file_size:
                raise ValueError("Invalid AVIF top-level box size.")
            top_level_boxes.append((pos, size, box_type, header_size, size_field))
            pos += size

        meta_box = next((box for box in top_level_boxes if box[2] == b"meta"), None)
        mdat_box = next((box for box in top_level_boxes if box[2] == b"mdat"), None)
        if meta_box is None or mdat_box is None or mdat_box[0] + mdat_box[1] != file_size or meta_box[0] > mdat_box[0]:
            raise ValueError("Unsupported AVIF file layout for metadata.")
        source.seek(meta_box[0])
        meta = source.read(meta_box[1])

    provisional_meta = _add_avif_exif_item(meta, 0, len(exif), 0)
    offset_delta = len(provisional_meta) - len(meta)
    exif_offset = mdat_box[0] + mdat_box[1] + offset_delta
    updated_meta = _add_avif_exif_item(meta, exif_offset, len(exif), offset_delta)

    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "wb") as destination, open(path, "rb") as source:
            for pos, size, box_type, header_size, size_field in top_level_boxes:
                source.seek(pos)
                if box_type == b"ftyp":
                    ftyp = bytearray(source.read(size))
                    # The frontend metadata reader recognizes the avif major brand; avis remains a compatible sequence brand.
                    if ftyp[8:12] == b"avis" and b"avif" in ftyp[16:]:
                        ftyp[8:12] = b"avif"
                    destination.write(ftyp)
                elif box_type == b"meta":
                    destination.write(updated_meta)
                elif box_type == b"moov":
                    destination.write(_adjust_avif_chunk_offsets(source.read(size), offset_delta))
                elif box_type == b"mdat":
                    new_size = size + len(exif)
                    if size_field == 1:
                        destination.write(struct.pack(">I4sQ", 1, b"mdat", new_size))
                    elif size_field == 0:
                        destination.write(struct.pack(">I4s", 0, b"mdat"))
                    elif new_size <= 0xFFFFFFFF:
                        destination.write(struct.pack(">I4s", new_size, b"mdat"))
                    else:
                        raise ValueError("AVIF media-data box exceeds its 32-bit size field.")
                    source.seek(pos + header_size)
                    _copy_file_bytes(source, destination, size - header_size)
                    destination.write(exif)
                else:
                    _copy_file_bytes(source, destination, size)
        os.chmod(temp_path, os.stat(path).st_mode)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_image(
    img_tensor: torch.Tensor,
    file_format: str,
    bit_depth: str,
    colorspace: str,
) -> bytes:
    """Encode a single HxWxC (or channel-less HxW grayscale) tensor to PNG or
    EXR bytes in memory. Grayscale is written as single-channel PNG / Y-only EXR.

    For EXR the input is interpreted according to `colorspace` and converted
    to scene-linear (EXR's convention) before writing:

      "sRGB"   → input is sRGB-encoded Rec. 709; apply inverse sRGB EOTF.
      "HDR"    → input is HLG-encoded Rec. 2020 (BT.2100); apply inverse HLG
                 OETF to get scene-linear, per BT.2100 Note 5a.
      "linear" → input is already scene-linear (Rec. 709 primaries); write
                 through unchanged. Use this for renderer/compositor output.

    For PNG, colorspace selection does not modify pixels — PNG is delivered
    sRGB-encoded and there is no PNG path for wide-gamut HDR in this node.
    """
    if img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(-1)  # Some nodes emit grayscale as (H, W) with no channel dim, mask-style.
    height, width, num_channels = img_tensor.shape

    spec = _FORMAT_SPECS.get((file_format, bit_depth, num_channels))
    if spec is None:
        raise ValueError(
            f"No {file_format}/{bit_depth} encoder for {num_channels}-channel images: "
            "supported channel counts are 1 (grayscale), 3 (RGB) and 4 (RGBA)."
        )

    if spec["dtype"] == np.float32:
        # EXR path: preserve full range, no clamp.
        if colorspace == "sRGB":
            img_tensor = srgb_to_linear(img_tensor)
        elif colorspace == "HDR":
            img_tensor = hlg_to_linear(img_tensor)
        img_np = img_tensor.cpu().numpy().astype(np.float32)
    else:
        # PNG path: quantize to integer range.
        scaled = (img_tensor * spec["scale"]).clamp(0, spec["scale"])
        img_np = scaled.to(torch.int32).cpu().numpy().astype(spec["dtype"])

    # Encode directly via CodecContext. PyAV's `image2` muxer does NOT write to
    # BytesIO (it expects a real file path), so we bypass the container entirely.
    # For single-frame PNG/EXR the raw codec output IS the file.
    codec = av.CodecContext.create(file_format, "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = spec["stream_fmt"]
    codec.time_base = Fraction(1, 1)

    frame = av.VideoFrame.from_ndarray(img_np, format=spec["frame_fmt"])
    if spec["frame_fmt"] != spec["stream_fmt"]:
        frame = frame.reformat(format=spec["stream_fmt"])
    frame.pts = 0
    frame.time_base = codec.time_base

    packets = list(codec.encode(frame)) + list(codec.encode(None))  # flush with None
    return b"".join(bytes(p) for p in packets)


def _set_avif_color_properties(target, colorspace: str) -> None:
    color_primaries, color_trc, yuv_colorspace = _AVIF_COLOR_PROPERTIES[colorspace]
    target.color_primaries = color_primaries
    target.color_trc = color_trc
    target.colorspace = yuv_colorspace
    target.color_range = ColorRange.MPEG


def _avif_frame(image: torch.Tensor, bit_depth: str, colorspace: str, pixel_format: str) -> av.VideoFrame:
    if image.ndim == 2:
        image = image.unsqueeze(-1)
    num_channels = image.shape[-1]
    if num_channels not in (1, 3):
        raise ValueError(
            "AVIF saving supports 1-channel grayscale and 3-channel RGB images; PyAV's SVT-AV1 encoder does not support alpha."
        )

    if bit_depth == "10-bit YUV420":
        image_np = (image * 65535.0).clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
        frame_format = "gray16le" if num_channels == 1 else "rgb48le"
    else:
        image_np = (image * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        frame_format = "gray" if num_channels == 1 else "rgb24"
    if num_channels == 1:
        image_np = image_np[..., 0]

    frame = av.VideoFrame.from_ndarray(image_np, format=frame_format)
    frame = frame.reformat(format=pixel_format, dst_colorspace=_AVIF_COLOR_PROPERTIES[colorspace][2])
    _set_avif_color_properties(frame, colorspace)
    return frame


def _save_avif(
    images: torch.Tensor,
    output_path: str,
    bit_depth: str,
    colorspace: str,
    crf: int,
    fps: float = 1.0,
    loop_count: int | None = None,
    metadata: dict | None = None,
) -> None:
    if bit_depth == "auto":
        bit_depth = "10-bit YUV420" if colorspace in ("HDR", "HDR PQ") else "8-bit YUV420"
    pixel_format = "yuv420p10le" if bit_depth == "10-bit YUV420" else "yuv420p"
    options = {"loop": str(loop_count)} if loop_count is not None else None

    with av.open(output_path, mode="w", format="avif", options=options) as container:
        frame_rate = Fraction(round(fps * 1000), 1000)
        stream = container.add_stream("libsvtav1", rate=frame_rate)
        stream.width = images.shape[2]
        stream.height = images.shape[1]
        stream.pix_fmt = pixel_format
        stream.options = {"crf": str(crf), "preset": "8"}
        _set_avif_color_properties(stream.codec_context, colorspace)

        for image in images:
            for packet in stream.encode(_avif_frame(image, bit_depth, colorspace, pixel_format)):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    if metadata:
        inject_avif_metadata(output_path, metadata)


class SaveImageAdvancedDestructive(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveImageAdvancedDestructive",
            search_aliases=["save", "save image", "export image", "output image", "write image"],
            display_name="Save Image (Advanced+Destructive)",
            description="Saves the input images to the ComfyUI output directory but clears the selected folder first",
            category="image",
            essentials_category="Basics",
            inputs=[
                IO.Image.Input("images", tooltip="The images to save."),
                IO.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=(
                        "The prefix for the file to save. May include formatting tokens such as %date:yyyy-MM-dd% or %Empty Latent Image.width%."
                    ),
                ),
                IO.DynamicCombo.Input(
                    "format",
                    options=[
                        IO.DynamicCombo.Option(
                            "png",
                            [
                                IO.Combo.Input("bit_depth", options=["8-bit", "16-bit"], default="8-bit", advanced=True),
                                IO.Combo.Input("input_color_space", options=["sRGB"], default="sRGB", advanced=True),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "exr",
                            [
                                IO.Combo.Input("bit_depth", options=["32-bit float"], default="32-bit float", advanced=True),
                                IO.Combo.Input(
                                    "input_color_space",
                                    options=["sRGB", "HDR", "linear"],
                                    default="sRGB",
                                    advanced=True,
                                    tooltip=(
                                        "Colorspace of the input tensor. The EXR is always written as scene-linear in the matching gamut.\n"
                                        "sRGB — input is sRGB-encoded Rec.709; the inverse sRGB EOTF is applied.\n"
                                        "HDR — input is HLG-encoded Rec.2020 (BT.2100); the inverse HLG OETF is applied to get scene-linear light.\n"
                                        "linear — input is already scene-linear (Rec.709 primaries); written through unchanged. Use this for renderer/compositor output."
                                    ),
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "avif",
                            [
                                IO.Combo.Input(
                                    "bit_depth",
                                    options=["auto", "8-bit YUV420", "10-bit YUV420"],
                                    default="auto",
                                    advanced=True,
                                    tooltip="Auto uses 8-bit YUV420 for sRGB and 10-bit YUV420 for HDR.",
                                ),
                                IO.Combo.Input(
                                    "input_color_space",
                                    options=["sRGB", "HDR", "HDR PQ"],
                                    default="sRGB",
                                    advanced=True,
                                    tooltip="Colorspace of the input images. HDR selects BT.2020/HLG and HDR PQ selects BT.2020/PQ.",
                                ),
                                IO.Int.Input(
                                    "crf",
                                    default=18,
                                    min=1,
                                    max=63,
                                    advanced=True,
                                    tooltip="Lower values produce higher quality and larger files.",
                                ),
                                IO.DynamicCombo.Input(
                                    "save_mode",
                                    display_name="save mode",
                                    options=[
                                        IO.DynamicCombo.Option("still images", []),
                                        IO.DynamicCombo.Option(
                                            "animated",
                                            [
                                                IO.Float.Input("fps", default=6.0, min=0.01, max=1000.0, step=0.01),
                                                IO.Int.Input(
                                                    "loop_count",
                                                    default=0,
                                                    min=0,
                                                    max=1000,
                                                    advanced=True,
                                                    tooltip="Number of times to loop the animation. 0 loops forever.",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    tooltip="The file format in which to save the image.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, filename_prefix: str, format: dict) -> IO.NodeOutput:
        file_format = format["format"]
        bit_depth = format["bit_depth"]
        colorspace = format.get("input_color_space", "sRGB")

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )

        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo
        write_metadata = not args.disable_metadata
        for filename in os.listdir(full_output_folder):
            file_path = os.path.join(full_output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
        counter = 0
        results = []
        if file_format == "avif":
            metadata = None
            if write_metadata:
                metadata = {}
                if prompt is not None:
                    metadata["prompt"] = prompt
                if extra_pnginfo:
                    metadata.update(extra_pnginfo)
            save_mode = format["save_mode"]
            animated = save_mode["save_mode"] == "animated"
            batches = [images] if animated else images.unsqueeze(1)
            for batch_number, batch in enumerate(batches):
                name = filename.replace("%batch_num%", str(batch_number))
                file = f"{name}_{counter:05}.avif"
                _save_avif(
                    batch,
                    os.path.join(full_output_folder, file),
                    bit_depth,
                    colorspace,
                    format["crf"],
                    fps=save_mode.get("fps", 1.0),
                    loop_count=save_mode.get("loop_count") if animated else None,
                    metadata=metadata,
                )
                results.append({"filename": file, "subfolder": subfolder, "type": "output"})
                counter += 1
            ui = {"images": results}
            if animated and len(images) > 1:
                ui["animated"] = (True,)
            return IO.NodeOutput(images, ui=ui)

        for batch_number, image in enumerate(images):
            encoded = _encode_image(image, file_format, bit_depth, colorspace)

            if write_metadata:
                if file_format == "png":
                    encoded = inject_png_metadata(encoded, prompt, extra_pnginfo)
                elif file_format == "exr":
                    encoded = inject_exr_metadata(encoded, prompt, extra_pnginfo, colorspace)

            name = filename.replace("%batch_num%", str(batch_number))
            file = f"{name}_{counter:05}.{file_format}"
            with open(os.path.join(full_output_folder, file), "wb") as f:
                f.write(encoded)

            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1

        return IO.NodeOutput(images, ui={"images": results})
