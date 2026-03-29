#!/usr/bin/env python3
"""
Render "Kellen" using Times New Roman font at different scales.

This module generates a visual comparison image showing the text "Kellen" rendered
at multiple scales with measurement annotations. It also extracts and overlays
text from PDF renders with character-level replacements.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import os
import json
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List, Any
import numpy.typing as npt

# Import from redaction_analysis package
from redaction_analysis.extraction import (
    RedactionBox,
    PDFSentenceExtractor,
    CharacterOverlay,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# PDF extraction regions
PDF_SECTION_Y0, PDF_SECTION_Y1 = 550, 650
PDF_SECTION_X0, PDF_SECTION_X1 = 100, 930

# Detection thresholds
MIN_BOX_WIDTH, MIN_BOX_HEIGHT = 60, 25
FILL_RATIO_THRESHOLD = 0.85
CHAR_MIN_WIDTH, CHAR_MIN_HEIGHT = 3, 8

# Character positioning
CHAR_E_POSITION_FROM_RIGHT = 5  # 'e' in "made"
CHAR_M_POSITION_FROM_RIGHT = 7  # 'm' in "made"

# Layout constants
SPACING_VERTICAL = 50
SPACING_HORIZONTAL = 10
BORDER_PADDING = 5

# Color constants
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (200, 200, 200)
COLOR_BLUE = (100, 100, 200)
COLOR_RED = (255, 0, 0)

# Overlay transparency
OVERLAY_ALPHA = 0.5

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class RenderConfig:
    """Configuration for rendering Kellen comparison image.
    
    Attributes:
        base_dir: Base directory for file I/O
        text: The text to render (default: "Kellen")
        font_size_large: Large font size in points (default: 48pt)
        font_size_small: Small font size in points (default: 12pt)
        scale_factor_4x: 4× display scale for 12pt text
        scale_factor_3125x: 3.125× scale matching PyMuPDF rendering
        min_canvas_width: Minimum canvas width in pixels
        canvas_extra_height: Extra height for layout spacing
        default_box_width_1_4: Default width for redaction box 1-4
        default_box_width_1_5: Default width for redaction box 1-5
        default_box_height: Default height for redaction boxes
        font_path: Path to Times New Roman font file
    """
    base_dir: str
    text: str = "Kellen"
    font_size_large: int = 48
    font_size_small: int = 12
    scale_factor_4x: int = 4
    scale_factor_3125x: float = 3.125
    min_canvas_width: int = 800
    canvas_extra_height: int = 850
    default_box_width_1_4: int = 119
    default_box_width_1_5: int = 131
    default_box_height: int = 53
    font_path: str = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"

    @property
    def output_path(self) -> str:
        """Get output path for generated comparison image."""
        return os.path.join(self.base_dir, "kellen_comparison.png")

    @property
    def boxes_path(self) -> str:
        """Get path to redaction boxes JSON file."""
        return os.path.join(self.base_dir, "redaction_boxes.json")

    @property
    def page_full_path(self) -> str:
        """Get path to full-page PDF render."""
        return os.path.join(self.base_dir, "page_full.png")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _load_font(font_path: str, font_size: int, fallback: Optional[ImageFont.FreeTypeFont] = None) -> ImageFont.FreeTypeFont:
    """Load TrueType font from path with fallback support.
    
    Args:
        font_path: Path to font file (.ttf or .otf)
        font_size: Font size in points
        fallback: Optional fallback font if loading fails
        
    Returns:
        Loaded font object or default font
        
    Raises:
        OSError: Only if font_path doesn't exist and no fallback provided
    """
    try:
        return ImageFont.truetype(font_path, font_size)
    except (OSError, IOError) as e:
        if fallback is not None:
            return fallback
        print(f"Warning: Could not load font {font_path}: {e}")
        return ImageFont.load_default()


def _render_text_image(text: str, font: ImageFont.FreeTypeFont, fill: Tuple[int, int, int]) -> Tuple[Image.Image, int, int]:
    """Render text to PIL Image with exact bounding box.
    
    Args:
        text: Text string to render
        font: PIL ImageFont object
        fill: RGB color tuple for text
        
    Returns:
        Tuple of (rendered_image, width_px, height_px)
    """
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    canvas = Image.new('RGB', (text_w, text_h), COLOR_WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=fill)
    return canvas, text_w, text_h


def _load_redaction_dimensions(config: RenderConfig) -> Tuple[int, int, int]:
    """Load redaction box dimensions from JSON file.
    
    Args:
        config: RenderConfig with paths and defaults
        
    Returns:
        Tuple of (box_width_1_4, box_width_1_5, box_height)
        Falls back to config defaults if file not found or corrupted
    """
    box_width_1_4 = config.default_box_width_1_4
    box_width_1_5 = config.default_box_width_1_5
    box_height = config.default_box_height

    try:
        with open(config.boxes_path, 'r') as file_obj:
            boxes = json.load(file_obj)

        for box in boxes:
            if box.get('uuid') == 'EFTA00037366-1-4':
                box_width_1_4 = box['width_pix']
                box_height = box['height_pix']
            elif box.get('uuid') == 'EFTA00037366-1-5':
                box_width_1_5 = box['width_pix']
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Info: Using default box dimensions ({e})")

    return box_width_1_4, box_width_1_5, box_height


def _draw_measurement_bar_and_height(
    canvas: npt.NDArray[np.uint8],
    image_y_offset: int,
    image_height: int,
    width_px: int,
    text_height_px: int,
    label: str,
    height_x: int,
    canvas_w: int,
    bar_x_start: int = 10,
) -> None:
    """Draw width measurement bar and height indicator for rendered text.
    
    Args:
        canvas: Output canvas to draw on
        image_y_offset: Y position of the image top edge
        image_height: Height of the image in pixels
        width_px: Width to measure with bar
        text_height_px: Text height for height indicator
        label: Label text for measurement bar
        height_x: X position for height indicator line
        canvas_w: Canvas width for bounds checking
        bar_x_start: X position to start measurement bar (default: 10)
    """
    bar_y = image_y_offset + image_height + 10
    x_start = bar_x_start
    x_end = min(canvas_w - 1, bar_x_start + width_px)

    # Draw width bar and end markers
    cv2.line(canvas, (x_start, bar_y), (x_end, bar_y), COLOR_BLACK, 2)
    cv2.line(canvas, (x_start, bar_y - 3), (x_start, bar_y + 3), COLOR_BLACK, 2)
    cv2.line(canvas, (x_end, bar_y - 3), (x_end, bar_y + 3), COLOR_BLACK, 2)
    cv2.putText(canvas, label, (x_end + 5, bar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1)

    # Draw height indicator
    cv2.line(canvas, (height_x, image_y_offset), (height_x, image_y_offset + text_height_px), COLOR_BLUE, 2)
    cv2.line(canvas, (height_x - 3, image_y_offset), (height_x + 3, image_y_offset), COLOR_BLUE, 1)
    cv2.line(canvas, (height_x - 3, image_y_offset + text_height_px), (height_x + 3, image_y_offset + text_height_px), COLOR_BLUE, 1)
    cv2.putText(canvas, f"h:{text_height_px}px", (height_x - 30, image_y_offset + text_height_px + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BLUE, 1)


def _draw_text_samples_with_measurements(
    canvas: npt.NDArray[np.uint8],
    img_48pt: Image.Image,
    img_12pt_4x: Image.Image,
    img_12pt_3125x: Image.Image,
    width_48pt: int,
    height_48pt: int,
    height_12pt: int,
    scale_4x: float,
    scale_3125x: float,
    canvas_w: int,
) -> int:
    """Draw three Kellen text samples with measurement bars and borders.
    
    Args:
        canvas: Output canvas to draw on
        img_48pt: 48pt font rendered image
        img_12pt_4x: 12pt font scaled 4× image
        img_12pt_3125x: 12pt font scaled 3.125× image
        width_48pt: Width of 48pt text in pixels
        height_48pt: Height of 48pt text in pixels
        height_12pt: Original height of 12pt text before scaling
        scale_4x: 4× scale factor
        scale_3125x: 3.125× scale factor
        canvas_w: Canvas width for positioning
    
    Returns:
        Y position after the last measurement bar
    """
    height_x = canvas_w - 40
    
    # First image (48pt font)
    y_offset = 10
    canvas[y_offset:y_offset+img_48pt.height, 10:10+img_48pt.width] = np.array(img_48pt)
    _draw_measurement_bar_and_height(canvas, y_offset, img_48pt.height, width_48pt, height_48pt,
                                     f"Kellen 48pt (font)  {width_48pt}px", height_x, canvas_w)
    
    # Second image (12pt scaled 4x)
    bar_y = y_offset + img_48pt.height + 10
    y_offset = bar_y + 25
    cv2.rectangle(canvas,
                  (10 - BORDER_PADDING, y_offset - BORDER_PADDING),
                  (10 + img_12pt_4x.width + BORDER_PADDING, y_offset + img_12pt_4x.height + BORDER_PADDING),
                  COLOR_GRAY, 1)
    canvas[y_offset:y_offset+img_12pt_4x.height, 10:10+img_12pt_4x.width] = np.array(img_12pt_4x)
    display_height_4x = int(height_12pt * scale_4x)
    _draw_measurement_bar_and_height(canvas, y_offset, img_12pt_4x.height, img_12pt_4x.width, display_height_4x,
                                     f"Kellen 12pt  {img_12pt_4x.width}px (shown {scale_4x}x)", height_x, canvas_w)
    
    # Third image (12pt scaled 3.125x)
    bar_y = y_offset + img_12pt_4x.height + 10
    y_offset = bar_y + 25
    cv2.rectangle(canvas,
                  (10 - BORDER_PADDING, y_offset - BORDER_PADDING),
                  (10 + img_12pt_3125x.width + BORDER_PADDING, y_offset + img_12pt_3125x.height + BORDER_PADDING),
                  COLOR_GRAY, 1)
    canvas[y_offset:y_offset+img_12pt_3125x.height, 10:10+img_12pt_3125x.width] = np.array(img_12pt_3125x)
    display_height_3125x = int(height_12pt * scale_3125x)
    _draw_measurement_bar_and_height(canvas, y_offset, img_12pt_3125x.height, img_12pt_3125x.width, display_height_3125x,
                                     f"Kellen 12pt  {img_12pt_3125x.width}px (shown {scale_3125x}x)", height_x, canvas_w)
    
    return y_offset + img_12pt_3125x.height + 10


def _draw_redaction_boxes_with_measurements(
    canvas: npt.NDArray[np.uint8],
    start_y: int,
    box_width_1_4: int,
    box_width_1_5: int,
    box_height: int,
    height_x: int,
) -> int:
    """Draw redaction boxes 1-4 and 1-5 with labels and measurement annotations.
    
    Args:
        canvas: Output canvas to draw on
        start_y: Y position to start drawing boxes
        box_width_1_4: Width of redaction box 1-4
        box_width_1_5: Width of redaction box 1-5
        box_height: Height of both redaction boxes
        height_x: X position for height indicator line
    
    Returns:
        Y position after the last measurement bar
    """
    box_y_start = start_y + 30
    
    # Draw redaction box 1-4
    cv2.rectangle(canvas, (10, box_y_start), (10 + box_width_1_4, box_y_start + box_height), COLOR_BLACK, -1)
    cv2.putText(canvas, "1-4", (15, box_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

    # Height indicator for box 1-4
    cv2.line(canvas, (height_x, box_y_start), (height_x, box_y_start + box_height), COLOR_BLUE, 2)
    cv2.line(canvas, (height_x - 3, box_y_start), (height_x + 3, box_y_start), COLOR_BLUE, 1)
    cv2.line(canvas, (height_x - 3, box_y_start + box_height), (height_x + 3, box_y_start + box_height), COLOR_BLUE, 1)
    cv2.putText(canvas, f"h:{box_height}px", (height_x - 30, box_y_start + box_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BLUE, 1)

    # Measurement bar for box 1-4
    bar_y_1_4 = box_y_start + box_height + 10
    x_end_1_4 = 10 + box_width_1_4
    cv2.line(canvas, (10, bar_y_1_4), (x_end_1_4, bar_y_1_4), COLOR_BLACK, 2)
    cv2.line(canvas, (10, bar_y_1_4 - 3), (10, bar_y_1_4 + 3), COLOR_BLACK, 1)
    cv2.line(canvas, (x_end_1_4, bar_y_1_4 - 3), (x_end_1_4, bar_y_1_4 + 3), COLOR_BLACK, 1)
    cv2.putText(canvas, f"1-4  {box_width_1_4}px",
                (x_end_1_4 + 5, bar_y_1_4 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1)

    # Draw redaction box 1-5
    box_1_5_y = bar_y_1_4 + 25
    cv2.rectangle(canvas, (10, box_1_5_y), (10 + box_width_1_5, box_1_5_y + box_height), COLOR_BLACK, -1)
    cv2.putText(canvas, "1-5", (15, box_1_5_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

    # Height indicator for box 1-5
    cv2.line(canvas, (height_x, box_1_5_y), (height_x, box_1_5_y + box_height), COLOR_BLUE, 2)
    cv2.line(canvas, (height_x - 3, box_1_5_y), (height_x + 3, box_1_5_y), COLOR_BLUE, 1)
    cv2.line(canvas, (height_x - 3, box_1_5_y + box_height), (height_x + 3, box_1_5_y + box_height), COLOR_BLUE, 1)
    cv2.putText(canvas, f"h:{box_height}px", (height_x - 30, box_1_5_y + box_height + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BLUE, 1)

    # Measurement bar for box 1-5
    bar_y_1_5 = box_1_5_y + box_height + 10
    x_end_1_5 = 10 + box_width_1_5
    cv2.line(canvas, (10, bar_y_1_5), (x_end_1_5, bar_y_1_5), COLOR_BLACK, 2)
    cv2.line(canvas, (10, bar_y_1_5 - 3), (10, bar_y_1_5 + 3), COLOR_BLACK, 1)
    cv2.line(canvas, (x_end_1_5, bar_y_1_5 - 3), (x_end_1_5, bar_y_1_5 + 3), COLOR_BLACK, 1)
    cv2.putText(canvas, f"1-5  {box_width_1_5}px",
                (x_end_1_5 + 5, bar_y_1_5 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1)
    
    return bar_y_1_5


# ============================================================================
# PDF PROCESSING CLASS
# ============================================================================


# ============================================================================
# CHARACTER OVERLAY CLASS
# ============================================================================


# ============================================================================
# MAIN RENDERING FUNCTIONS
# ============================================================================

def render_kellen_comparison() -> None:
    """Render 'Kellen' using Times New Roman font at different scales with measurement bars."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = RenderConfig(base_dir=base_dir)

    text = config.text
    # From PDF vol00008-official-doj-latest-efta00037366.pdf, line 7 measurements:
    # Contact at 16pt effective (12pt × 96/72 scaling): 155×35px
    # Scaling Kellen to match Contact's visual size: use 48pt
    # This makes Kellen visually comparable to Contact
    font = _load_font(config.font_path, config.font_size_large)
    img2, width2_px, text2_height = _render_text_image(text, font, (0, 0, 0))
    
    # Method 3: Render from Times New Roman font at 12pt
    font_12pt = _load_font(config.font_path, config.font_size_small, fallback=font)
    img3, _, text3_height = _render_text_image(text, font_12pt, (0, 0, 0))
    
    # Scale up 12pt version 4x for visibility (but keep original measurements)
    scale_factor_4x = config.scale_factor_4x
    img3_display_4x = cv2.resize(np.array(img3), 
                                  (img3.width * scale_factor_4x, img3.height * scale_factor_4x), 
                                  interpolation=cv2.INTER_NEAREST)
    img3_display_4x = Image.fromarray(img3_display_4x)
    
    # Scale up 12pt version 3.125x (matches PyMuPDF scale)
    scale_factor_3125x = config.scale_factor_3125x
    img3_display_3125x = cv2.resize(np.array(img3), 
                                     (int(img3.width * scale_factor_3125x), int(img3.height * scale_factor_3125x)), 
                                     interpolation=cv2.INTER_NEAREST)
    img3_display_3125x = Image.fromarray(img3_display_3125x)
    
    # Render letter 'e' at 12pt scaled 4x (for overlay on 'made')
    img_e, _, _ = _render_text_image("e", font_12pt, (255, 0, 0))
    img_e_display_4x = cv2.resize(np.array(img_e), 
                                   (img_e.width * scale_factor_4x, img_e.height * scale_factor_4x), 
                                   interpolation=cv2.INTER_NEAREST)
    img_e_display_4x = Image.fromarray(img_e_display_4x)
    
    # Render letter 'm' at 12pt (for overlay on 'made')
    img_m, _, _ = _render_text_image("m", font_12pt, (255, 0, 0))
    img_m_display_3125x = Image.fromarray(np.array(img_m))
    
    # Combine images with measurement bars
    max_w = max(img2.width, img3_display_4x.width, img3_display_3125x.width)
    canvas_h = img2.height + img3_display_4x.height + img3_display_3125x.height + config.canvas_extra_height  # Space for 3 Kellens (font-based) + 2 redaction boxes + Contact comparison + spacing
    # Ensure canvas is wide enough for "Contact was made with " text + redaction box 1-4
    # At 16pt with 3.125x scale, "Contact was made with " is approximately 420px, plus box_width_1_4 (119px) = ~550px
    canvas_w = max(max_w + 50, config.min_canvas_width)  # Wider canvas to accommodate full text + box
    
    final_canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    
    # Draw the three Kellen text samples with measurements
    bar_y4 = _draw_text_samples_with_measurements(
        final_canvas, img2, img3_display_4x, img3_display_3125x,
        width2_px, text2_height, text3_height,
        scale_factor_4x, scale_factor_3125x, canvas_w
    )
    
    # Draw redaction boxes with measurements
    box_width_1_4, box_width_1_5, box_height = _load_redaction_dimensions(config)
    height_x = canvas_w - 40
    bar_y_1_5 = _draw_redaction_boxes_with_measurements(
        final_canvas, bar_y4, box_width_1_4, box_width_1_5, box_height, height_x
    )

    # Extract and overlay the "Contact was made with" sentence from PDF
    pdf_img = cv2.imread(config.page_full_path)
    if pdf_img is not None:
        # Position for the extracted sentence section
        contact_y_start = bar_y_1_5 + SPACING_VERTICAL
        contact_x_start = SPACING_HORIZONTAL
        contact_line_y = contact_y_start

        try:
            # Initialize PDF extractor
            extractor = PDFSentenceExtractor()
            
            # Detect redaction boxes and extract text band
            boxes = extractor.extract_redaction_boxes(pdf_img)
            section_extracted, box_positions, same_row = extractor.crop_text_band(pdf_img, boxes)
            
            if section_extracted is not None and box_positions is not None:
                first_box_x, first_box_y, first_box_w, first_box_h = box_positions

            if section_extracted is not None and box_positions is not None:
                first_box_x, first_box_y, first_box_w, first_box_h = box_positions

                # Overlay green "Kellen" on first redaction box
                if first_box_w > 4 and first_box_h > 4:
                    kellen_ref = np.array(img3_display_4x)
                    kellen_gray = cv2.cvtColor(kellen_ref, cv2.COLOR_RGB2GRAY)
                    
                    # Crop to tight bounding box
                    _, kellen_mask = cv2.threshold(kellen_gray, 250, 255, cv2.THRESH_BINARY_INV)
                    nz = cv2.findNonZero(kellen_mask)
                    if nz is not None:
                        xk, yk, wk, hk = cv2.boundingRect(nz)
                        kellen_gray = kellen_gray[yk:yk+hk, xk:xk+wk]
                    
                    overlay_h, overlay_w = kellen_gray.shape[:2]
                    
                    # Measure text height for proper scaling
                    text_left_region = section_extracted[:, :max(1, first_box_x)]
                    target_height = overlay_h
                    if text_left_region.size > 0:
                        text_gray = cv2.cvtColor(text_left_region, cv2.COLOR_BGR2GRAY)
                        _, text_mask = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY_INV)
                        rows_nonzero = np.where(np.any(text_mask > 0, axis=1))[0]
                        if rows_nonzero.size > 0:
                            target_height = rows_nonzero.max() - rows_nonzero.min() + 1
                    
                    # Scale to match text height
                    if target_height != overlay_h:
                        scale_factor = target_height / overlay_h
                        new_w = int(overlay_w * scale_factor)
                        new_h = int(overlay_h * scale_factor)
                        kellen_gray = cv2.resize(kellen_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        overlay_h, overlay_w = new_h, new_w
                    
                    alpha_k = (255 - kellen_gray).astype(np.uint8)
                    
                    # Position with baseline alignment
                    place_x = first_box_x + 3
                    place_y = first_box_y + (first_box_h - overlay_h) // 2
                    
                    if text_left_region.size > 0:
                        text_gray = cv2.cvtColor(text_left_region, cv2.COLOR_BGR2GRAY)
                        _, text_mask = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY_INV)
                        rows_nonzero = np.where(np.any(text_mask > 0, axis=1))[0]
                        if rows_nonzero.size > 0:
                            text_bottom = int(rows_nonzero.max())
                            place_y = text_bottom - overlay_h - 1
                    
                    # Clip and blend
                    x0, y0 = max(0, place_x), max(0, place_y)
                    x1 = min(section_extracted.shape[1], place_x + overlay_w)
                    y1 = min(section_extracted.shape[0], place_y + overlay_h)
                    
                    if x1 > x0 and y1 > y0:
                        ax0, ay0 = x0 - place_x, y0 - place_y
                        ax1, ay1 = ax0 + (x1 - x0), ay0 + (y1 - y0)
                        
                        alpha_sub = alpha_k[ay0:ay1, ax0:ax1].astype(np.float32) / 255.0
                        region = section_extracted[y0:y1, x0:x1].astype(np.float32)
                        
                        # Green overlay
                        region[:, :, 0] *= (1.0 - alpha_sub)
                        region[:, :, 2] *= (1.0 - alpha_sub)
                        region[:, :, 1] = np.clip(region[:, :, 1] + alpha_sub * 255.0, 0, 255)
                        
                        section_extracted[y0:y1, x0:x1] = region.astype(np.uint8)
                
                # Find all character boxes for overlay positioning
                char_boxes = extractor.find_character_boxes(section_extracted[:, :first_box_x])
                
                # Overlay light blue 'e' on the 'e' in "made"
                e_overlay = CharacterOverlay(img_e_display_4x, (135, 206, 250), OVERLAY_ALPHA)
                e_target = e_overlay.find_target_position(char_boxes, CHAR_E_POSITION_FROM_RIGHT, first_box_x - 80)
                
                if e_target is not None:
                    ex, ey, ew, eh = e_target
                    # Calculate centered position
                    e_place_x = ex + (ew - e_overlay.cropped_img.shape[1]) // 2
                    
                    # Baseline align
                    text_region = section_extracted[:, :first_box_x]
                    text_gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
                    _, text_mask = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY_INV)
                    rows_with_text = np.any(text_mask > 0, axis=1)
                    if rows_with_text.any():
                        text_bottom = int(np.where(rows_with_text)[0].max())
                        e_place_y = text_bottom - e_overlay.cropped_img.shape[0] - 2
                    else:
                        e_place_y = ey
                    
                    # Scale to match character height
                    target_size = (e_overlay.cropped_img.shape[1], eh) if eh > 0 else None
                    e_overlay.blend_onto(section_extracted, (e_place_x, e_place_y), target_size)
                
                # Overlay red 'm' on the 'm' in "made"
                m_overlay = CharacterOverlay(img_m_display_3125x, COLOR_RED, OVERLAY_ALPHA)
                m_target = m_overlay.find_target_position(char_boxes, CHAR_M_POSITION_FROM_RIGHT, first_box_x - 100)
                
                if m_target is not None and len(char_boxes) >= CHAR_M_POSITION_FROM_RIGHT:
                    mx, my, mw, mh = m_target
                    refined_x, refined_y, refined_w, refined_h = mx, my, mw, mh
                    
                    # Refine 'm' bbox if merged with 'a'
                    median_w = float(np.median([w for _, _, w, _ in char_boxes]))
                    if mw > 1.45 * median_w:
                        text_region = section_extracted[:, :first_box_x]
                        text_gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
                        _, text_mask = cv2.threshold(text_gray, 200, 255, cv2.THRESH_BINARY_INV)
                        roi = text_mask[my:my+mh, mx:mx+mw]
                        if roi.size > 0:
                            col_ink = np.sum(roi > 0, axis=0)
                            search_l = max(1, int(mw * 0.30))
                            search_r = min(mw - 1, int(mw * 0.85))
                            if search_r > search_l + 1:
                                split_rel = search_l + int(np.argmin(col_ink[search_l:search_r]))
                                left_roi = roi[:, :split_rel]
                                ys, xs = np.where(left_roi > 0)
                                if ys.size > 0:
                                    minx, maxx = int(xs.min()), int(xs.max())
                                    miny, maxy = int(ys.min()), int(ys.max())
                                    refined_x = mx + minx
                                    refined_y = my + miny
                                    refined_w = maxx - minx + 1
                                    refined_h = maxy - miny + 1
                                    if refined_w < m_overlay.cropped_img.shape[1] - 1:
                                        refined_w = m_overlay.cropped_img.shape[1]
                    
                    # Blend red 'm' at exact refined position
                    m_overlay.blend_onto(section_extracted, (refined_x, refined_y), (refined_w, refined_h))

                    # Blend red 'm' at exact refined position
                    m_overlay.blend_onto(section_extracted, (refined_x, refined_y), (refined_w, refined_h))
                
                # Place extracted section onto canvas with measurements
                required_width = contact_x_start + section_extracted.shape[1] + 140
                needed_height = contact_line_y + section_extracted.shape[0] + 40
                needed_width = max(final_canvas.shape[1], required_width)
                
                if needed_height > final_canvas.shape[0] or needed_width > final_canvas.shape[1]:
                    new_canvas = np.full(
                        (max(needed_height, final_canvas.shape[0]), needed_width, 3),
                        255, dtype=np.uint8
                    )
                    new_canvas[:final_canvas.shape[0], :final_canvas.shape[1], :] = final_canvas
                    final_canvas = new_canvas
                
                # Copy section onto canvas
                final_canvas[
                    contact_line_y:contact_line_y + section_extracted.shape[0],
                    contact_x_start:contact_x_start + section_extracted.shape[1]
                ] = section_extracted
                
                # Draw height measurement indicator
                section_height_x = min(
                    final_canvas.shape[1] - 20,
                    contact_x_start + section_extracted.shape[1] + 25
                )
                cv2.line(
                    final_canvas,
                    (section_height_x, contact_line_y),
                    (section_height_x, contact_line_y + section_extracted.shape[0]),
                    COLOR_BLUE, 2
                )
                cv2.line(
                    final_canvas,
                    (section_height_x - 3, contact_line_y),
                    (section_height_x + 3, contact_line_y),
                    COLOR_BLUE, 1
                )
                cv2.line(
                    final_canvas,
                    (section_height_x - 3, contact_line_y + section_extracted.shape[0]),
                    (section_height_x + 3, contact_line_y + section_extracted.shape[0]),
                    COLOR_BLUE, 1
                )
                cv2.putText(
                    final_canvas,
                    f"h:{section_extracted.shape[0]}px",
                    (section_height_x - 42, contact_line_y + section_extracted.shape[0] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_BLUE, 1
                )
                
                # Draw width measurement bar
                bar_y_combined = contact_line_y + section_extracted.shape[0] + SPACING_HORIZONTAL
                cv2.line(
                    final_canvas,
                    (contact_x_start, bar_y_combined),
                    (contact_x_start + section_extracted.shape[1], bar_y_combined),
                    COLOR_BLACK, 2
                )
                cv2.putText(
                    final_canvas,
                    f"section: {section_extracted.shape[1]}x{section_extracted.shape[0]}px",
                    (contact_x_start, bar_y_combined + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_BLACK, 1
                )
                
                max_y_needed = bar_y_combined + 25
            else:
                max_y_needed = contact_line_y + 40

        except Exception:
            pass
    
    # Save
    cv2.imwrite(config.output_path, cv2.cvtColor(final_canvas, cv2.COLOR_RGB2BGR))
    
    print(f"✓ Saved kellen_comparison.png ({final_canvas.shape[0]}×{final_canvas.shape[1]} pixels)")

if __name__ == "__main__":
    render_kellen_comparison()
