"""Image loading and validation module for TerraSentinel AI."""
import io
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

# Security bound against DecompressionBombError / memory exhaustion
Image.MAX_IMAGE_PIXELS = 50_000_000

MAX_DIMENSION = 2500  # Conservative safety bound for web interface

def load_image(
    source: Union[str, Path, bytes, io.BytesIO],
    max_dim: int = 1600
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image from path or bytes, normalizes it to uint8 RGB numpy array,

    and extracts structural metadata.

    Returns:
        tuple: (rgb_array, metadata_dict)
    Raises:
        ValueError: If file is malformed, unsupported, or exceeds safe limits.
    """
    try:
        if isinstance(source, (str, Path)):
            pil_img = Image.open(str(source))
        elif isinstance(source, bytes):
            pil_img = Image.open(io.BytesIO(source))
        elif hasattr(source, "read"):
            pil_img = Image.open(source)
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")

        # Validate PIL image
        pil_img.verify()
        # Re-open after verify (PIL requirement)
        if isinstance(source, (str, Path)):
            pil_img = Image.open(str(source))
        elif isinstance(source, bytes):
            pil_img = Image.open(io.BytesIO(source))
        else:
            source.seek(0)
            pil_img = Image.open(source)

        orig_format = pil_img.format or "UNKNOWN"
        orig_width, orig_height = pil_img.size
        orig_mode = pil_img.mode

        if orig_width == 0 or orig_height == 0:
            raise ValueError("Image dimensions cannot be zero.")

        # Convert palette images or CMYK or Grayscale or RGBA to RGB
        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            # Composite onto neutral black or white background
            background = Image.new("RGB", pil_img.size, (0, 0, 0))
            rgba_img = pil_img.convert("RGBA")
            background.paste(rgba_img, mask=rgba_img.split()[3])
            pil_img = background
        elif pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Check dimension limits
        scale_factor = 1.0
        if max(orig_width, orig_height) > max_dim:
            scale_factor = max_dim / float(max(orig_width, orig_height))
            new_w = max(1, int(orig_width * scale_factor))
            new_h = max(1, int(orig_height * scale_factor))
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        arr = np.array(pil_img, dtype=np.uint8)

        metadata = {
            "original_width": orig_width,
            "original_height": orig_height,
            "processed_width": arr.shape[1],
            "processed_height": arr.shape[0],
            "channels": arr.shape[2] if len(arr.shape) == 3 else 1,
            "original_mode": orig_mode,
            "format": orig_format,
            "rescaled": scale_factor < 1.0,
            "scale_factor": scale_factor,
        }

        return arr, metadata

    except Exception as e:
        raise ValueError(f"Image decoding failed: {str(e)}")


def harmonize_dimensions(
    img_before: np.ndarray,
    img_after: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Ensures both images have identical dimensions for pixel-wise and structural comparison.

    Resizes to the smaller or common bounding shape using high-fidelity interpolation.
    Returns:
        (harmonized_before, harmonized_after, was_resized)
    """
    h_b, w_b = img_before.shape[:2]
    h_a, w_a = img_after.shape[:2]

    if (h_b, w_b) == (h_a, w_a):
        return img_before, img_after, False

    # Common target resolution: min dimensions to avoid inventing high frequency detail
    target_w = min(w_b, w_a)
    target_h = min(h_b, h_a)

    pil_b = Image.fromarray(img_before).resize((target_w, target_h), Image.Resampling.LANCZOS)
    pil_a = Image.fromarray(img_after).resize((target_w, target_h), Image.Resampling.LANCZOS)

    return np.array(pil_b), np.array(pil_a), True
