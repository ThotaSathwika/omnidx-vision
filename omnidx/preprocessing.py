"""Modality-specific image-processing pipelines.

X-ray, MRI, CT, ultrasound and dermoscopy images have different noise and contrast
characteristics, so each modality gets its own chain of steps. All images are RGB uint8.
"""
from functools import partial

import cv2
import numpy as np


class ImageValidationError(ValueError):
    pass


def load_image(path_or_bytes):
    """Read a file path or raw bytes into an RGB uint8 array."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(path_or_bytes, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageValidationError("File is not a readable image (PNG/JPEG expected).")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def validate(img, modality):
    """Reject unusable images; return warnings for suspicious ones."""
    h, w = img.shape[:2]
    if min(h, w) < 64:
        raise ImageValidationError(f"Image too small ({w}x{h}); at least 64x64 is required.")
    if img.std() < 2:
        raise ImageValidationError("Image is blank or almost uniform.")
    warnings = []
    colour = np.abs(img[..., 0].astype(int) - img[..., 2].astype(int)).mean()
    if modality == "dermoscopy" and colour < 3:
        warnings.append("Image looks grayscale; dermoscopy images are normally colour.")
    if modality in ("xray", "mri", "ct", "ultrasound") and colour > 15:
        warnings.append(f"Image looks like a colour photo, not a {modality} scan.")
    return warnings


# ---------- individual steps (each takes and returns a uint8 image) ----------

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img


def resize_pad(img, size):
    """Resize keeping aspect ratio, then pad to size x size (no distortion)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    top, left = (size - h) // 2, (size - w) // 2
    return cv2.copyMakeBorder(img, top, size - h - top, left, size - w - left, cv2.BORDER_CONSTANT, value=0)


def median(img, k=3):
    return cv2.medianBlur(img, k)


def gaussian(img, k=3):
    return cv2.GaussianBlur(img, (k, k), 0)


def speckle_reduce(img):
    """Edge-preserving smoothing for ultrasound speckle noise."""
    return cv2.bilateralFilter(img, 7, 40, 7)


def clahe(img, clip=2.0, tiles=8):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    if img.ndim == 2:
        return c.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = c.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def percentile_normalize(img, lo=1, hi=99):
    """Stretch intensities between the 1st and 99th percentile to 0-255."""
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        return img
    return np.clip((img.astype(np.float32) - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)


def brain_extract(img):
    """Simple skull-stripping: keep the largest bright connected region."""
    from .segmentation import brain_mask
    return cv2.bitwise_and(img, img, mask=brain_mask(img))


def hair_removal(img):
    """DullRazor-style hair removal: black-hat filter finds hairs, inpainting fills them."""
    gray = to_gray(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, hair, 3, cv2.INPAINT_TELEA)


def color_constancy(img, power=6):
    """Shades-of-Gray colour constancy: removes lighting colour casts between cameras."""
    f = img.astype(np.float32)
    illum = np.power(np.mean(np.power(f, power), axis=(0, 1)), 1 / power)
    illum = illum / (np.linalg.norm(illum) + 1e-6) * np.sqrt(3)
    return np.clip(f / (illum + 1e-6), 0, 255).astype(np.uint8)


def pipeline(modality, size):
    """Ordered (name, function) steps for a modality."""
    r = partial(resize_pad, size=size)
    steps = {
        "xray": [("grayscale", to_gray), ("resize", r), ("denoise (median)", median),
                 ("CLAHE", clahe), ("normalise", percentile_normalize)],
        "mri": [("grayscale", to_gray), ("resize", r), ("denoise (gaussian)", gaussian),
                ("brain extraction", brain_extract), ("normalise", percentile_normalize), ("CLAHE", clahe)],
        "ct": [("grayscale", to_gray), ("resize", r), ("normalise", percentile_normalize),
               ("CLAHE", clahe)],
        "ultrasound": [("grayscale", to_gray), ("resize", r), ("speckle reduction", speckle_reduce),
                       ("CLAHE", partial(clahe, clip=1.5))],
        "dermoscopy": [("resize", r), ("hair removal", hair_removal),
                       ("colour constancy", color_constancy), ("CLAHE", partial(clahe, clip=1.5))],
    }
    if modality not in steps:
        raise ValueError(f"No pipeline for modality '{modality}'")
    return steps[modality]


def preprocess(img, modality, size=224, keep_steps=False):
    """Run the modality pipeline. Returns the RGB result (and the intermediate steps)."""
    history = [("original", img)]
    out = img
    for name, fn in pipeline(modality, size):
        out = fn(out)
        if keep_steps:
            history.append((name, out))
    out = to_rgb(out)
    return (out, history) if keep_steps else out
