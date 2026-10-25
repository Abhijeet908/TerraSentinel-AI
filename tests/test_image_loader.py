"""Tests for image loader and validation module."""
import numpy as np
from PIL import Image
import io
from modules.image_loader import load_image, harmonize_dimensions

def test_load_valid_rgb_image():
    # Create simple RGB image
    img = Image.new("RGB", (100, 80), color=(120, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    arr, meta = load_image(buf)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (80, 100, 3)
    assert meta["original_width"] == 100
    assert meta["original_height"] == 80
    assert meta["channels"] == 3


def test_load_rgba_image_composited():
    # Create RGBA image with transparent parts
    img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    arr, meta = load_image(buf)
    assert arr.shape == (50, 50, 3)
    assert meta["original_mode"] == "RGBA"


def test_invalid_malformed_image():
    # Corrupt random bytes should raise ValueError
    corrupt_bytes = b"NOT_AN_IMAGE_RANDOM_GARBAGE_BYTES_12345"
    load_image(corrupt_bytes)


def test_harmonize_different_dimensions():
    img1 = np.zeros((100, 150, 3), dtype=np.uint8)
    img2 = np.zeros((200, 300, 3), dtype=np.uint8)

    h1, h2, resized = harmonize_dimensions(img1, img2)
    assert resized is True
    assert h1.shape == h2.shape
    assert h1.shape == (100, 150, 3)


def test_harmonize_identical_dimensions():
    img1 = np.zeros((120, 120, 3), dtype=np.uint8)
    img2 = np.zeros((120, 120, 3), dtype=np.uint8)

    h1, h2, resized = harmonize_dimensions(img1, img2)
    assert resized is False
    assert h1.shape == img1.shape
