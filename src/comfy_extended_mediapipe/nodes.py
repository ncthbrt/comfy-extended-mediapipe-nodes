import torch

from .nodes_mediapipe import LoadMediaPipeExtendedFaceLandmarker, MediaPipeExtendedFaceMask


class ImageMinMax:
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("min", "max")
    FUNCTION = "min_max"

    def min_max(self, images):
        min = torch.amin(images[..., 0])
        max = torch.amax(images[..., 0])
        return (min, max)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ImageMinMax": ImageMinMax,
    "MediaPipeExtendedFaceMask": MediaPipeExtendedFaceMask,
    "LoadMediaPipeExtendedFaceLandmarker": LoadMediaPipeExtendedFaceLandmarker,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMinMax": "Image Min/Max",
    "MediaPipeExtendedFaceMask": "Mediapipe Extended Face Mask",
    "LoadMediaPipeExtendedFaceLandmarker": "Load Mediapipe Extended Face Landmarker",
}
