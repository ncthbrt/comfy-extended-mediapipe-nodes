"""Top-level package for comfy-extended-mediapipe."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Natalie Cuthbert"""
__email__ = "natalie@cuthbert.co.za"
__version__ = "0.0.1"

from .src.comfy_extended_mediapipe.nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./web"
