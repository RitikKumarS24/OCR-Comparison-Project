import os
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load a BGR image from disk with validation."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return img


def resize_upscale(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """Upscale low-resolution images using bicubic interpolation."""
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    new_dim = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_dim, interpolation=cv2.INTER_CUBIC)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR color image to single-channel grayscale."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise_gaussian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply gentle Gaussian smoothing to suppress high-frequency noise."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def denoise_median(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Median filtering to remove impulse noise while preserving text edges."""
    return cv2.medianBlur(gray, ksize)


def enhance_contrast_clahe(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Enhance local foreground-background contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sharpen_unsharp_mask(gray: np.ndarray) -> np.ndarray:
    """Restore crisp character boundaries using unsharp masking."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    return cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)


def deskew_image(gray: np.ndarray) -> np.ndarray:
    """Estimate text skew angle via minimum-area bounding box and straighten."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 20:
        return gray

    xy_coords = coords[:, ::-1].astype(np.float32)
    rect = cv2.minAreaRect(xy_coords)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < 0.5:
        return gray

    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_v = np.abs(rot_mat[0, 0])
    sin_v = np.abs(rot_mat[0, 1])
    new_w = int((h * sin_v) + (w * cos_v))
    new_h = int((h * cos_v) + (w * sin_v))
    rot_mat[0, 2] += (new_w / 2.0) - center[0]
    rot_mat[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        gray, rot_mat, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )


def threshold_otsu(gray: np.ndarray) -> np.ndarray:
    """Binarize image automatically using Otsu's inter-class variance maximization."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """Apply morphological opening to remove tiny background specks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def preprocess_pipeline(image_path: str, mode: str = "standard") -> np.ndarray:
    """
    Unified preprocessing pipeline returning a 3-channel BGR image compatible
    with all 5 OCR engines.
    Modes:
      - 'standard' : Grayscale -> Median Denoise -> CLAHE -> Deskew -> Sharpen
      - 'binary'   : Standard + Otsu Binarization + Morphological Cleanup
      - 'low_res'  : 1.8x Bicubic Upscale -> Denoise -> CLAHE -> Sharpen
    """
    img = load_image(image_path)

    if mode == "low_res":
        img = resize_upscale(img, scale=1.8)

    gray = to_grayscale(img)
    denoised = denoise_median(gray, ksize=3)
    contrasted = enhance_contrast_clahe(denoised, clip_limit=2.0)
    deskewed = deskew_image(contrasted)
    sharpened = sharpen_unsharp_mask(deskewed)

    if mode == "binary":
        otsu = threshold_otsu(sharpened)
        final_gray = morphological_cleanup(otsu)
    else:
        final_gray = sharpened

    return cv2.cvtColor(final_gray, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_cases = [
        ("rotated", "eng_01.png", "standard"),
        ("noisy", "eng_03.png", "binary"),
        ("blurry", "hin_01.png", "standard"),
        ("low_resolution", "eng_07.png", "low_res"),
    ]

    for folder, fname, mode in test_cases:
        in_path = os.path.join(project_root, "dataset", folder, fname)
        out_img = preprocess_pipeline(in_path, mode=mode)
        out_path = os.path.join(project_root, "results", f"preview_{folder}_{fname}")
        cv2.imwrite(out_path, out_img)
        print(f"Processed [{folder}/{fname}] with mode='{mode}' -> Saved {out_path}")