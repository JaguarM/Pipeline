#!/usr/bin/env python3
"""
reveal_redacted_chars.py
========================
For every PDF in PDFS/, render page 0 at 3.125x zoom, detect solid-black
redaction boxes across the full page, and for every character whose bbox
intersects a redaction box overlay it in blue Times New Roman -- using the
identical _render_text_image + CharacterOverlay.blend_onto pipeline as the
'm' overlay in kellen_render.py.

The image is kept in RGB throughout so PIL colours are correct; it is only
converted to BGR at save time.

Output: redacted_char_outputs/<stem>_revealed.png
"""

import os
import numpy as np
import fitz
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple

from redaction_analysis.extraction import CharacterOverlay

ZOOM          = 3.125
FONT_PATH     = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
BLUE_RGB      = (0, 0, 255)
OVERLAY_ALPHA = 1.0

REDACT_MIN_W  = 20
REDACT_MIN_H  = 8
REDACT_FILL   = 0.85

_font_cache: Dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size_px: int) -> ImageFont.FreeTypeFont:
    if size_px not in _font_cache:
        try:
            _font_cache[size_px] = ImageFont.truetype(FONT_PATH, size_px)
        except (OSError, IOError):
            _font_cache[size_px] = ImageFont.load_default()
    return _font_cache[size_px]


def _render_text_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
) -> Optional[Image.Image]:
    """Render text tightly cropped to its glyph bbox on a white RGB background.
    Mirrors kellen_render._render_text_image exactly."""
    try:
        bb = font.getbbox(text)
    except Exception:
        return None
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    if w <= 0 or h <= 0:
        return None
    img = Image.new("RGB", (w, h), (255, 255, 255))
    ImageDraw.Draw(img).text((-bb[0], -bb[1]), text, font=font, fill=fill)
    return img


def detect_redaction_boxes(img_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < REDACT_MIN_W or h < REDACT_MIN_H:
            continue
        area = w * h
        if area == 0:
            continue
        fill = float(np.count_nonzero(binary[y:y+h, x:x+w])) / float(area)
        if fill >= REDACT_FILL:
            boxes.append((x, y, w, h))
    return boxes


def rects_intersect(
    ax0: float, ay0: float, ax1: float, ay1: float,
    bx0: int, by0: int, bx1: int, by1: int,
) -> bool:
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def process_pdf(pdf_path: str, out_path: str, page_num: int = 0) -> None:
    doc  = fitz.open(pdf_path)
    page = doc[page_num]

    mat     = fitz.Matrix(ZOOM, ZOOM)
    pix     = page.get_pixmap(matrix=mat)
    img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3).copy()

    redact_boxes = detect_redaction_boxes(img_rgb)
    print(f"  -> {len(redact_boxes)} redaction box(es) detected")
    if not redact_boxes:
        cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        doc.close()
        return

    text_dict = page.get_text("rawdict")
    revealed  = 0

    for block in text_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                font_size_pt = span.get("size", 10.0)
                size_px = max(4, round(font_size_pt * ZOOM))
                font    = _get_font(size_px)

                for char in span.get("chars", []):
                    c    = char["c"]
                    bbox = char["bbox"]

                    cx0 = bbox[0] * ZOOM;  cy0 = bbox[1] * ZOOM
                    cx1 = bbox[2] * ZOOM;  cy1 = bbox[3] * ZOOM
                    cw  = max(1, int(round(cx1 - cx0)))
                    ch  = max(1, int(round(cy1 - cy0)))

                    hit = any(
                        rects_intersect(cx0, cy0, cx1, cy1,
                                        rx, ry, rx + rw, ry + rh)
                        for (rx, ry, rw, rh) in redact_boxes
                    )
                    if not hit:
                        continue

                    # -- Same method as 'm' in kellen_render --
                    # Step A: render glyph with PIL in blue on white (RGB)
                    rendered = _render_text_image(c, font, BLUE_RGB)
                    if rendered is None:
                        continue

                    # Step B: build CharacterOverlay
                    #   mirrors: m_overlay = CharacterOverlay(img_m, COLOR_RED, OVERLAY_ALPHA)
                    overlay = CharacterOverlay(rendered, BLUE_RGB, OVERLAY_ALPHA)

                    # Step C: blend onto RGB canvas using the character bbox,
                    # but preserve the rendered glyph aspect ratio instead of
                    # stretching it to the full bbox width. Align to the bbox
                    # bottom so the line position stays stable.
                    glyph_h, glyph_w = overlay.cropped_img.shape[:2]
                    if glyph_h <= 0 or glyph_w <= 0:
                        continue

                    scale = ch / float(glyph_h)
                    target_h = max(1, int(round(glyph_h * scale)))
                    target_w = max(1, int(round(glyph_w * scale)))
                    place_x = int(round(cx0 + (cw - target_w) / 2.0))
                    place_y = int(round(cy1 - target_h))

                    overlay.blend_onto(
                        img_rgb,
                        (place_x, place_y),
                        target_size=(target_w, target_h),
                    )
                    revealed += 1

    print(f"  -> {revealed} character(s) revealed in blue")
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    doc.close()


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir  = os.path.join(base_dir, "PDFS")
    out_dir  = os.path.join(base_dir, "redacted_char_outputs")
    os.makedirs(out_dir, exist_ok=True)

    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    print(f"Found {len(pdfs)} PDF(s)\n")

    for pdf_file in pdfs:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        stem     = os.path.splitext(pdf_file)[0]
        out_path = os.path.join(out_dir, f"{stem}_revealed.png")
        print(f"=== {pdf_file} ===")
        try:
            process_pdf(pdf_path, out_path)
            print(f"  saved -> {out_path}\n")
        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
