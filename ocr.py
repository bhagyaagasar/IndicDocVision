"""OCR pipeline for scanned pages (Kannada / English / Hindi).

Steps: load image or PDF page -> optional enhancement (CLAHE, denoise, deskew,
adaptive binarization for degraded scans) -> Tesseract -> text + confidence.

Environment variables
  OCR_LANGS       default "kan+eng+hin" (only installed languages are used)
  TESSERACT_CMD   full path to tesseract.exe (Windows, if not on PATH)
  POPPLER_PATH    folder containing pdftoppm.exe (Windows, for scanned PDFs)
"""
import os

import cv2
import numpy as np
import pytesseract
from PIL import Image

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

DEFAULT_LANGS = os.getenv("OCR_LANGS", "kan+eng+hin")
POPPLER_PATH = os.getenv("POPPLER_PATH") or None
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class OCRSetupError(RuntimeError):
    pass


def installed_langs():
    try:
        return set(pytesseract.get_languages(config=""))
    except Exception as e:  # tesseract missing or not on PATH
        raise OCRSetupError(
            "Tesseract OCR was not found. Install it and add it to PATH, or set "
            "TESSERACT_CMD to the full path of the tesseract executable. "
            f"({e})"
        )


def resolve_langs(requested=DEFAULT_LANGS):
    """Return (lang_string_for_tesseract, [missing_langs])."""
    want = [l.strip() for l in requested.split("+") if l.strip()]
    have = installed_langs()
    use = [l for l in want if l in have]
    missing = [l for l in want if l not in have]
    if not use:
        if "eng" not in have:
            raise OCRSetupError("No requested OCR language is installed.")
        use = ["eng"]
    return "+".join(use), missing


# ---------------------------------------------------------------- preprocessing
def _estimate_skew(gray, max_angle=10.0, step=0.5):
    """Projection-profile method: the angle whose rotation gives the sharpest
    row-sum profile is the one that aligns the text lines."""
    h, w = gray.shape
    scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    sh, sw = bw.shape
    center = (sw / 2, sh / 2)
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D(center, float(a), 1.0)
        rot = cv2.warpAffine(bw, M, (sw, sh), flags=cv2.INTER_NEAREST)
        score = np.var(rot.sum(axis=1, dtype=np.float64))
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def deskew(gray):
    angle = _estimate_skew(gray)
    if abs(angle) < 0.4:
        return gray, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return out, angle


def preprocess(img_rgb, enhance=False):
    """enhance=False: light touch (clean scans). enhance=True: for old, faded,
    low-contrast or noisy pages (CLAHE + denoise + adaptive binarization)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if gray.shape[1] < 1000:  # small image: upscale for Tesseract
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray, _ = deskew(gray)
    if enhance:
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15
        )
    return gray


# ------------------------------------------------------------------------- OCR
def ocr_image(img_rgb, langs, enhance=False):
    """Returns (text, mean_confidence_0_to_100)."""
    proc = preprocess(img_rgb, enhance)
    data = pytesseract.image_to_data(
        proc, lang=langs, config="--oem 1 --psm 3", output_type=pytesseract.Output.DICT
    )
    lines, confs, order = {}, [], []
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(word)
        c = float(data["conf"][i])
        if c >= 0:
            confs.append(c)
    out, prev = [], None
    for key in order:
        if prev is not None and key[:2] != prev[:2]:
            out.append("")  # blank line between paragraphs
        out.append(" ".join(lines[key]))
        prev = key
    mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
    return "\n".join(out).strip(), mean_conf


def _load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def ocr_file(path, langs, enhance=False, dpi=300):
    """OCR an image or PDF. Returns a list of {page, text, confidence, method}."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        text, conf = ocr_image(_load_image(path), langs, enhance)
        return [{"page": 1, "text": text, "confidence": conf, "method": "ocr"}]
    if ext != ".pdf":
        raise ValueError(f"Unsupported file type: {ext}")

    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        layer = (page.extract_text() or "").strip()
        if len(layer) >= 30:  # PDF already has a text layer: no OCR needed
            pages.append({"page": i, "text": layer, "confidence": 100.0, "method": "text-layer"})
            continue
        from pdf2image import convert_from_path

        try:
            imgs = convert_from_path(
                path, dpi=dpi, first_page=i, last_page=i, poppler_path=POPPLER_PATH
            )
        except Exception as e:
            raise OCRSetupError(
                "Could not rasterize the PDF page. Install Poppler and add it to PATH "
                f"(or set POPPLER_PATH). ({e})"
            )
        text, conf = ocr_image(np.array(imgs[0].convert("RGB")), langs, enhance)
        pages.append({"page": i, "text": text, "confidence": conf, "method": "ocr"})
    return pages
