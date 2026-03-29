#!/usr/bin/env python3
import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any, Optional


PDF_PATH = "./PDFS/vol00008-official-doj-latest-efta00037366.pdf"
OUT_CLEAN = "./contact_line_native_clean.png"
OUT_OVERLAY = "./contact_line_native_ms_red_overlay.png"
TARGET_TEXT = "Contact was made with"
FONT_PATHS = [
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    "./times.ttf",
]


def find_target_chars(page: Any, target_text: str) -> Optional[List[Dict[str, Any]]]:
    text_dict = page.get_text("rawdict")
    chars = []

    for block in text_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                size = span.get("size", 12)
                for char in span.get("chars", []):
                    chars.append(
                        {
                            "char": char["c"],
                            "bbox": char["bbox"],
                            "origin": char.get("origin", (0, 0)),
                            "font": font,
                            "size": size,
                        }
                    )

    text_stream = "".join(c["char"] for c in chars)
    idx = text_stream.find(target_text)
    if idx == -1:
        return None
    return chars[idx : idx + len(target_text)]


def load_ms_times(size: float) -> Tuple[ImageFont.FreeTypeFont, str]:
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, int(round(size))), path
        except Exception:
            continue
    raise RuntimeError("Could not load Microsoft Times font from known paths")


def main():
    doc = fitz.open(PDF_PATH)
    page = doc[0]

    # Render at native PDF scale (no scaling)
    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    page_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    base = Image.fromarray(page_img, mode="RGB")

    target_chars = find_target_chars(page, TARGET_TEXT)
    if not target_chars:
        raise RuntimeError(f"Target text not found: {TARGET_TEXT!r}")

    x0 = min(c["bbox"][0] for c in target_chars)
    y0 = min(c["bbox"][1] for c in target_chars)
    x1 = max(c["bbox"][2] for c in target_chars)
    y1 = max(c["bbox"][3] for c in target_chars)

    pad_x = 4
    pad_y = 3
    crop_box = (
        max(0, int(np.floor(x0)) - pad_x),
        max(0, int(np.floor(y0)) - pad_y),
        min(base.width, int(np.ceil(x1)) + pad_x),
        min(base.height, int(np.ceil(y1)) + pad_y),
    )

    clean_crop = base.crop(crop_box)
    clean_crop.save(OUT_CLEAN)

    font_size = target_chars[0]["size"]
    font, loaded_path = load_ms_times(font_size)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    phrase = "".join(c["char"] for c in target_chars)

    # Draw each character at its own PDF baseline origin.
    # PDF rawdict origin is the text baseline point; in Pillow, anchor="ls"
    # means left + baseline, which aligns with that coordinate system.
    y_offset_px = 1
    for i, char_info in enumerate(target_chars):
        ch = char_info["char"]
        ox, oy = char_info["origin"]
        x_offset_px = 1 if i < 4 else 0
        draw.text((ox + x_offset_px, oy + y_offset_px), ch, font=font, fill=(255, 0, 0, 128), anchor="ls")

    composited = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    overlay_crop = composited.crop(crop_box)
    overlay_crop.save(OUT_OVERLAY)

    doc.close()

    print(f"Target phrase: {phrase}")
    print(f"Native page size: {base.width}x{base.height}")
    print(f"Loaded font: {loaded_path} @ {int(round(font_size))}pt")
    print(f"Clean output: {OUT_CLEAN}")
    print(f"Overlay output: {OUT_OVERLAY}")


if __name__ == "__main__":
    main()