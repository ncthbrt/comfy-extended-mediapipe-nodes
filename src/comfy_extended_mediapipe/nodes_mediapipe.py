"""Extended ComfyUI nodes for the pure-PyTorch MediaPipe Face Landmarker port.

Custom IO types:
  FACE_LANDMARKS   — {"frames": List[List[face_dict]], "image_size": (H, W),
                      "connection_sets": dict[str, frozenset[(int, int)]]}
                     face_dict: bbox_xyxy, blendshapes, landmarks_xy,
                                landmarks_3d, presence, score, transformation_matrix
"""

from sre_compile import CATEGORY

import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths
import numpy as np
import torch
from comfy_api.latest import io
from comfy_extras.mediapipe.face_landmarker import FaceLandmarker
from comfy_extras.nodes_mediapipe import FaceDetectionType, FaceLandmarksType
from PIL import Image, ImageDraw

MOUTH_INDICES = [
    (308, 415),
    (415, 310),
    (310, 311),
    (311, 312),
    (312, 13),
    (13, 82),
    (82, 81),
    (81, 80),
    (80, 191),
    (191, 78),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (308, 291),
]


EYEBROW_INDICES = [
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    (336, 285),
    (285, 295),
    (295, 282),
    (282, 283),
    (283, 276),
    (276, 300),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    (107, 55),
    (55, 65),
    (65, 52),
    (52, 53),
    (53, 46),
    (46, 70),
]

EYEBROW_CONTOURS_INDICES = [
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    (336, 300),
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (285, 276),
    (107, 66),
    (66, 105),
    (105, 63),
    (63, 70),
    (70, 107),
    (55, 65),
    (65, 52),
    (52, 53),
    (53, 46),
    (46, 55),
]


NOSTRIL_INDICES = [
    (102, 49),
    (49, 48),
    (48, 115),
    (115, 102),
    (331, 279),
    (279, 278),
    (278, 344),
    (344, 331),
]

IRIS_CENTER_INDICES = [(468, 468), (473, 473)]

_CANONICAL_KEYS = ("canonical_vertices", "procrustes_indices", "procrustes_weights")
_CONTOUR_PARTS = (
    "face_oval",
    "left_eye",
    "right_eye",
    "left_eyebrow",
    "right_eyebrow",
    "lips",
    "nostrils",
)


# Topology keys unioned by the 'all' connections preset (contour parts + irises + nose).
_ALL_CONNECTION_PARTS: tuple[str, ...] = (
    *_CONTOUR_PARTS,
    "irises",
    "nose",
    "mouth",
    "iris_centers",
    "eyebrows",
    "eyebrow_lines",
    "nostrils",
)

_CUSTOM_FEATURES: tuple[tuple[str, bool], ...] = (
    ("face_oval", True),
    ("lips", True),
    ("left_eye", True),
    ("right_eye", True),
    ("left_eyebrow", True),
    ("right_eyebrow", True),
    ("irises", True),
    ("nose", True),
    ("tesselation", False),
    ("mouth", True),
    ("iris_centers", True),
    ("eyebrows", True),
    ("eyebrow_lines", True),
    ("nostrils", True),
)

# Mask region presets — closed-loop topologies only.
_MASK_REGIONS: tuple[str, ...] = (
    "face_oval",
    "lips",
    "left_eye",
    "right_eye",
    "irises",
    "mouth",
    "iris_centers",
    "eyebrows",
    "nostrils",
)

_MASK_CUSTOM_FEATURES: tuple[tuple[str, bool], ...] = (
    ("face_oval", True),
    ("lips", False),
    ("left_eye", False),
    ("right_eye", False),
    ("irises", False),
    ("mouth", False),
    ("iris_centers", False),
    ("eyebrows", False),
    ("eyebrow_lines", False),
    ("nostrils", False),
)


class FaceLandmarkerExtendedModel:
    """Loaded FaceLandmarker variants + ModelPatcher per variant.

    Safetensors layout: `detector_short.*` / `detector_full.*` plus shared
    `mesh.*`, `blendshapes.*`, `canonical_*`, and `topology.*`.
    PReLU forces plain-nn / fp32 (manual_cast strands buffers across devices).
    """

    def __init__(self, state_dict: dict):
        self.load_device = comfy.model_management.text_encoder_device()
        offload_device = comfy.model_management.text_encoder_offload_device()
        self.dtype = torch.float32

        # FACEMESH_* connection sets, embedded as int32 (N, 2) under topology.*.
        base: dict[str, frozenset] = {}
        for k in [k for k in state_dict if k.startswith("topology.")]:
            base[k[len("topology.") :]] = frozenset(map(tuple, state_dict.pop(k).tolist()))
        base["mouth"] = frozenset(map(tuple, MOUTH_INDICES))
        base["iris_centers"] = frozenset(map(tuple, IRIS_CENTER_INDICES))
        base["eyebrows"] = frozenset(map(tuple, EYEBROW_INDICES))
        base["eyebrow_lines"] = frozenset(map(tuple, EYEBROW_CONTOURS_INDICES))
        base["nostrils"] = frozenset(map(tuple, NOSTRIL_INDICES))
        base["contours"] = frozenset().union(*(base[p] for p in _CONTOUR_PARTS))
        base["all"] = (
            base["contours"]
            | base["irises"]
            | base["nose"]
            | base["mouth"]
            | base["iris_centers"]
            | base["eyebrows"]
            | base["nostrils"]
            | base["eyebrow_lines"]
        )

        self.connection_sets: dict[str, frozenset] = base
        self.canonical_data: dict[str, np.ndarray] = {k: state_dict.pop(k).numpy() for k in _CANONICAL_KEYS}

        shared = {k: v for k, v in state_dict.items() if k.startswith(("mesh.", "blendshapes."))}

        self.models: dict[str, FaceLandmarker] = {}
        self.patchers: dict[str, comfy.model_patcher.ModelPatcher] = {}
        for variant in ("short", "full"):
            prefix = f"detector_{variant}."
            sub = dict(shared)
            sub.update({f"detector.{k[len(prefix) :]}": v for k, v in state_dict.items() if k.startswith(prefix)})
            fl = FaceLandmarker(
                device=offload_device,
                dtype=self.dtype,
                operations=None,
                detector_variant=variant,
            ).eval()
            fl.load_state_dict(sub, strict=False)

            self.models[variant] = fl
            self.patchers[variant] = comfy.model_patcher.CoreModelPatcher(
                fl,
                load_device=self.load_device,
                offload_device=offload_device,
                size=comfy.model_management.module_size(fl),
            )

    def detect_batch(self, images, num_faces: int, score_thresh: float, variant: str):
        comfy.model_management.load_model_gpu(self.patchers[variant])
        return self.models[variant].detect_batch(images, num_faces=num_faces, score_thresh=score_thresh)


class LoadMediaPipeExtendedFaceLandmarker(io.ComfyNode):
    """Load MediaPipe Face Landmarker v2 weights. Contains both detector variants
    (short / full), shared mesh, blendshapes, and canonical geometry."""

    CATEGORY = "model/loads"
    FUNCTION = "execute"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadMediaPipeExtendedFaceLandmarker",
            search_aliases=[
                "face",
                "facial",
                "mediapipe",
                "face landmark",
                "face mesh",
                "blazeface",
                "face detection",
            ],
            display_name="Load Extended Face Detection Model (MediaPipe)",
            category="model/loaders",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=folder_paths.get_filename_list("detection"),
                    tooltip="Face detection model from models/detection/.",
                ),
            ],
            outputs=[FaceDetectionType.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("detection", model_name), safe_load=True)
        wrapper = FaceLandmarkerExtendedModel(sd)
        return io.NodeOutput(wrapper)


class Ring:
    edges: list[int]

    def __init__(self, edges: list[int]) -> None:
        self.edges = edges

    def draw(self, lmks: dict[tuple[int, int], float | int], image_draw: ImageDraw.ImageDraw):
        image_draw.polygon([(float(lmks[i, 0]), float(lmks[i, 1])) for i in self.edges], fill=255)


class Point:
    point: int

    def __init__(self, point: int) -> None:
        self.point = point

    def draw(self, lmks: dict[tuple[int, int], float | int], image_draw: ImageDraw.ImageDraw):
        image_draw.point((float(lmks[self.point, 0]), float(lmks[self.point, 1])), fill=255)


class Ellipse:
    corners: list[int]

    def __init__(self, corners: list[int]) -> None:
        self.corners = corners

    def draw(self, lmks: dict[tuple[int, int], float | int], image_draw: ImageDraw.ImageDraw):
        min_x = 1000000
        max_x = -1000000
        min_y = 1000000
        max_y = -1000000
        for i in self.corners:
            pos = (float(lmks[i, 0]), float(lmks[i, 1]))
            if pos[0] < min_x:
                min_x = pos[0]
            if pos[0] > max_x:
                max_x = pos[0]
            if pos[1] < min_y:
                min_y = pos[1]
            if pos[1] > max_y:
                max_y = pos[1]
        image_draw.ellipse((min_x, min_y, max_x, max_y), fill=255)


class EyebrowLine:
    top_contour: list[int]
    bottom_contour: list[int]

    def __init__(self, top_contour: list[int], bottom_contour: list[int]) -> None:
        self.top_contour = top_contour
        self.bottom_contour = bottom_contour

    def draw(self, lmks: dict[tuple[int, int], float | int], image_draw: ImageDraw.ImageDraw):
        points: list[tuple[float, float]] = []
        last = len(self.top_contour) - 1
        print("LEN TOP: ", len(self.top_contour), " LEN BOT: ", len(self.bottom_contour), "\n")
        average_point = (0.0, 0.0)
        for i in range(0, len(self.top_contour)):
            if i == 0 or i == last:
                offset = 1
                if i == last:
                    offset = -1
                top_index_0 = self.top_contour[i]
                bottom_index_0 = self.bottom_contour[i]
                top_index_1 = self.top_contour[i + offset]
                bottom_index_1 = self.bottom_contour[i + offset]
                top_point_0 = (float(lmks[top_index_0, 0]), float(lmks[top_index_0, 1]))
                bottom_point_0 = (float(lmks[bottom_index_0, 0]), float(lmks[bottom_index_0, 1]))
                top_point_1 = (float(lmks[top_index_1, 0]), float(lmks[top_index_1, 1]))
                bottom_point_1 = (float(lmks[bottom_index_1, 0]), float(lmks[bottom_index_1, 1]))
                average_point = (
                    (top_point_0[0] + bottom_point_0[0] + top_point_1[0] + bottom_point_1[0]) / 4.0,
                    (top_point_0[1] + bottom_point_0[1] + top_point_1[1] + bottom_point_1[1]) / 4.0,
                )
                points.append(average_point)
            else:
                top_index = self.top_contour[i]
                bottom_index = self.bottom_contour[i]
                top_point = (float(lmks[top_index, 0]), float(lmks[top_index, 1]))
                bottom_point = (float(lmks[bottom_index, 0]), float(lmks[bottom_index, 1]))
                average_point: tuple[float, float] = (
                    (top_point[0] + bottom_point[0]) / 2.0,
                    (top_point[1] + bottom_point[1]) / 2.0,
                )
                points.append(average_point)
        image_draw.line(points, fill=255, width=1)


def _ordered_rings(edges: frozenset[tuple[int, int]], key: str) -> list[Point | Ring | Ellipse | EyebrowLine]:
    """Walk an unordered edge set into one or more closed-loop vertex rings
    (handles multi-loop sets like FACEMESH_LIPS: outer + inner)."""
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited: set[int] = set()
    rings: list[Point | Ring | Ellipse | EyebrowLine] = []
    eyebrow_contours: list[list[int]] = []
    if key == "eyebrow_lines":
        return [
            EyebrowLine([300, 293, 334, 296, 336], [276, 283, 282, 295, 285]),
            EyebrowLine([107, 66, 105, 63, 70], [55, 65, 52, 53, 46]),
        ]
    for start in adj:
        if start in visited:
            continue
        ring = [start]
        visited.add(start)
        prev, cur = -1, start
        while True:
            nxt = next((v for v in adj[cur] if v != prev), None)
            if nxt is None or nxt == start:
                break
            ring.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        if len(ring) == 1:
            rings.append(Point(ring[0]))
        elif key == "irises":
            rings.append(Ellipse(ring))
        elif key == "eyebrow_lines":
            eyebrow_contours.append(ring)
            if len(eyebrow_contours) == 2:
                rings.append(EyebrowLine(eyebrow_contours[0], eyebrow_contours[1]))
                eyebrow_contours.clear()
        else:
            rings.append(Ring(ring))
    return rings


class MediaPipeExtendedFaceMask(io.ComfyNode):
    """Binary mask from face landmarks, filled polygon per face. One mask per
    frame in the batch; faces in the same frame composite (union)."""

    CATEGORY = "image/detection"
    FUNCTION = "execute"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MediaPipeExtendedFaceMask",
            search_aliases=[
                "face",
                "facial",
                "mediapipe",
                "face mask",
                "blazeface",
                "face detection",
                "visualize",
            ],
            display_name="Draw Extended Face Mask (MediaPipe)",
            category="image/detection",
            description="Draws a mask from face landmarks.",
            inputs=[
                FaceLandmarksType.Input("face_landmarks"),
                io.DynamicCombo.Input(
                    "regions",
                    tooltip="'all' = union of face_oval+lips+eyes+irises (which collapses to face_oval since it encloses the rest). 'custom' = toggle each region individually for combos like lips+eyes.",
                    options=[
                        io.DynamicCombo.Option("all", []),
                        io.DynamicCombo.Option(
                            "custom",
                            [
                                io.Boolean.Input(
                                    reg,
                                    default=default,
                                    tooltip=f"Include the '{reg}' region in the mask.",
                                )
                                for reg, default in _MASK_CUSTOM_FEATURES
                            ],
                        ),
                    ],
                ),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(cls, face_landmarks, regions) -> io.NodeOutput:
        sets = face_landmarks["connection_sets"]
        sel = regions["regions"]
        if sel == "custom":
            picked = [reg for reg, _ in _MASK_CUSTOM_FEATURES if regions.get(reg, False)]
        else:
            picked = list(_MASK_REGIONS)
        rings = []
        rings = [r for reg in picked for r in _ordered_rings(sets[reg], reg)]
        frames = face_landmarks["frames"]
        H, W = face_landmarks["image_size"]
        masks = np.zeros((len(frames), H, W), dtype=np.uint8)
        pbar = comfy.utils.ProgressBar(len(frames))
        for bi, per_frame in enumerate(frames):
            if per_frame:
                pil = Image.new("L", (W, H), 0)
                draw = ImageDraw.Draw(pil)
                for f in per_frame:
                    lmks = f["landmarks_xy"]
                    for ring in rings:
                        ring.draw(lmks, draw)
                masks[bi] = np.asarray(pil)
            pbar.update_absolute(bi + 1)
        return io.NodeOutput(
            torch.from_numpy(masks)
            .to(
                device=comfy.model_management.intermediate_device(),
                dtype=comfy.model_management.intermediate_dtype(),
            )
            .div_(255.0)
        )
