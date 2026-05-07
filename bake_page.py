"""
bake_page.py — Standalone CLI tool

Extracts a page from a PDF, applies the redaction-recovery bake, and saves
the result as a PNG.

Usage:
    python bake_page.py <pdf_path> <page> [output.png] [--strength S]

Arguments:
    pdf_path    Path to the PDF file
    page        Page number (1-based)
    output      Output PNG path (default: <pdf>_p<page>_baked.png)
    --strength  Bake strength 0.0–1.0 (default: 1.0)
"""

import argparse
import os
import sys

import cv2
import fitz
import numpy as np
from io import BytesIO
from PIL import Image


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def get_grayscale_image_bytes(doc, page_index, image_index=0):
    try:
        image_list = doc.get_page_images(page_index)
        if not image_list or image_index >= len(image_list):
            return None
        xref = image_list[image_index][0]
        pix = fitz.Pixmap(doc, xref)

        if pix.n > 1 or (pix.colorspace and pix.colorspace.name != fitz.csGRAY.name):
            try:
                pix = fitz.Pixmap(fitz.csGRAY, pix)
            except Exception:
                pass

        return pix.tobytes("png")
    except Exception as e:
        print(f"Error extracting image from page {page_index}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Mask building
# ---------------------------------------------------------------------------

def _dilate(m):
    d = m.copy()
    d[1:]  |= m[:-1]
    d[:-1] |= m[1:]
    d[:, 1:]  |= m[:, :-1]
    d[:, :-1] |= m[:, 1:]
    return d


def _remove_circles(rendered, black_mask):
    blurred = cv2.GaussianBlur(rendered, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1, minDist=30,
        param1=100, param2=30,
        minRadius=8, maxRadius=20,
    )
    if circles is None:
        return black_mask

    reject = np.zeros(rendered.shape, dtype=np.uint8)
    for cx, cy, r in np.round(circles[0]).astype(int):
        cv2.circle(reject, (cx, cy), r + 2, 1, thickness=cv2.FILLED)
    return black_mask & (reject == 0)


def _filter_components(black_mask):
    img = black_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = np.zeros_like(img)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 17 or h < 10:
            continue
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0 and area / perimeter < 2:
            continue
        cv2.drawContours(result, [cnt], -1, 255, thickness=cv2.FILLED)

    return result.astype(bool)


def _apply_edge_lines(m, border1, border2, outer1, black_mask, rendered):
    h_border1 = border1 & (
        np.roll(black_mask, 1, axis=0) | np.roll(black_mask, -1, axis=0)
    )
    v_border1 = border1 & ~h_border1

    h_border2 = border2 & (
        np.roll(outer1, 1, axis=0) | np.roll(outer1, -1, axis=0)
    )
    v_border2 = border2 & ~h_border2

    for h_border in (h_border1, h_border2):
        for y in range(h_border.shape[0]):
            row = h_border[y]
            if not np.any(row):
                continue
            padded = np.concatenate(([False], row, [False]))
            diff = np.diff(padded.astype(np.int8))
            for sx, ex in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
                m[y, sx:ex] = 255 - int(rendered[y, sx:ex].max())

    for v_border in (v_border1, v_border2):
        for x in range(v_border.shape[1]):
            col = v_border[:, x]
            if not np.any(col):
                continue
            padded = np.concatenate(([False], col, [False]))
            diff = np.diff(padded.astype(np.int8))
            for sy, ey in zip(np.where(diff == 1)[0], np.where(diff == -1)[0]):
                m[sy:ey, x] = 255 - int(rendered[sy:ey, x].max())


def build_mask_array(rendered):
    """Returns a uint8 mask (255=redacted interior, 0=clear) or None if no redactions."""
    black_mask = rendered <= 0
    black_mask = _remove_circles(rendered, black_mask)
    black_mask = _filter_components(black_mask)
    if not np.any(black_mask):
        return None

    outer1  = _dilate(black_mask)
    border1 = outer1 & ~black_mask
    border2 = _dilate(outer1) & ~outer1

    m = np.zeros(rendered.shape, dtype=np.uint8)
    m[black_mask] = 255
    _apply_edge_lines(m, border1, border2, outer1, black_mask, rendered)
    return m


# ---------------------------------------------------------------------------
# Bake (WebGL shader replica)
# ---------------------------------------------------------------------------

def bake_page(page_img: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    page = page_img.astype(np.float32) / 255.0
    m    = mask.astype(np.float32) / 255.0

    interior     = m > (254.5 / 255.0)
    blend_factor = np.maximum(1.0 - m * strength, 0.001)
    recovered    = np.minimum(page / blend_factor, 1.0)
    result       = np.where(interior, strength, recovered)

    return (result * 255.0).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract a PDF page, apply redaction-recovery bake, save as PNG."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("page", type=int, help="Page number (1-based)")
    parser.add_argument("output", nargs="?", help="Output PNG path (optional)")
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="Bake strength 0.0–1.0 (default: 1.0)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.pdf):
        sys.exit(f"Error: file not found: {args.pdf}")

    if args.strength < 0.0 or args.strength > 1.0:
        sys.exit("Error: --strength must be between 0.0 and 1.0")

    page_index = args.page - 1

    with open(args.pdf, "rb") as f:
        pdf_bytes = f.read()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        sys.exit(f"Error opening PDF: {e}")

    if page_index < 0 or page_index >= len(doc):
        doc.close()
        sys.exit(f"Error: page {args.page} out of range (PDF has {len(doc)} pages)")

    print(f"Extracting page {args.page} of {len(doc)}…")
    img_bytes = get_grayscale_image_bytes(doc, page_index)
    doc.close()

    if not img_bytes:
        sys.exit("Error: could not extract image from page")

    with Image.open(BytesIO(img_bytes)) as pil_img:
        page_arr = np.array(pil_img.convert("L"))

    print("Building redaction mask…")
    mask = build_mask_array(page_arr)

    if mask is not None:
        print(f"Mask found — baking with strength={args.strength}")
        baked = bake_page(page_arr, mask, args.strength)
        final_img = Image.fromarray(baked, "L")
    else:
        print("No redactions detected — saving original image")
        final_img = Image.fromarray(page_arr, "L")

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(args.pdf)[0]
        out_path = f"{base}_p{args.page}_baked.png"

    final_img.save(out_path, format="PNG")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
