"""
Module for overlaying names on redaction boxes in PDF documents.

This module provides functionality to create visual overlays of matched names
on redaction boxes using a glyph bank for character rendering.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import fitz
from PIL import Image


def _lookup_pair_gap(
    ch1: str,
    ch2: str,
    line_pair_map: Dict[str, float],
    page_pair_map: Dict[str, float]
) -> float:
    """Look up kerning between two characters."""
    pair_key = f"{ch1}{ch2}"
    
    # Try line-specific first
    if pair_key in line_pair_map:
        return line_pair_map[pair_key]
    
    # Fall back to page-wide
    if pair_key in page_pair_map:
        return page_pair_map[pair_key]
    
    # Default spacing
    return 0.0


def _render_name_with_glyphs(
    name: str,
    glyph_bank: Dict[str, Any],
    spacing_model: Dict[str, Any],
    target_center_y_pt: Optional[float],
    start_x_px: int,
    start_y_px: int
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Render a name using glyph bank characters with learned kerning.
    
    Returns list of (glyph_alpha, x_pos, y_pos) tuples.
    """
    line_pair_map = {}
    line_space_px = None
    
    # Find nearest line model based on y position
    if spacing_model.get("line_models") and target_center_y_pt is not None:
        nearest = min(
            spacing_model["line_models"],
            key=lambda lm: abs(lm["y_center"] - float(target_center_y_pt))
        )
        line_pair_map = nearest.get("pair_map", {})
        line_space_px = nearest.get("space_px")
    
    if line_space_px is None:
        line_space_px = spacing_model.get("page_space_px", 8.0)
    
    # First pass: collect glyphs to determine alignment
    temp_glyphs = []
    current_x = float(start_x_px)
    prev_char = None
    
    for ch in name:
        if ch == " ":
            current_x += float(line_space_px)
            prev_char = None
            continue
        
        # Get glyph (try exact case, then swapcase)
        entry = glyph_bank.get(ch) or glyph_bank.get(ch.swapcase())
        if entry is None or "alpha" not in entry:
            prev_char = ch
            continue
        
        glyph_alpha = entry["alpha"]
        glyph_h, glyph_w = glyph_alpha.shape[:2]
        baseline_ratio = entry.get("baseline_ratio", 0.80)
        
        # Add kerning from previous character
        if prev_char is not None:
            page_pair_map = spacing_model.get("page_pair_map", {})
            kern = _lookup_pair_gap(prev_char, ch, line_pair_map, page_pair_map)
            current_x += kern
        
        temp_glyphs.append((glyph_alpha, int(current_x), glyph_h, baseline_ratio))
        
        current_x += glyph_w
        prev_char = ch
    
    # Use a common baseline position for all glyphs
    common_baseline_y = start_y_px
    
    # Second pass: position all glyphs aligned to common baseline
    glyphs_to_render = []
    for glyph_alpha, glyph_x, glyph_h, baseline_ratio in temp_glyphs:
        # Position each glyph so its baseline aligns with the common baseline
        baseline_offset = baseline_ratio * glyph_h
        glyph_y = int(common_baseline_y - baseline_offset)
        glyphs_to_render.append((glyph_alpha, glyph_x, glyph_y))
    
    return glyphs_to_render


def overlay_names_on_redaction_boxes(
    matched_pairs: Dict[Tuple, Dict[str, Any]],
    pdf_path: str,
    glyph_bank: Dict[str, Any],
    spacing_model: Optional[Dict[str, Any]] = None,
    output_dir: str = "./overlays",
    dpi: int = 300,
    visualize_each: bool = False
) -> List[str]:
    """
    Create overlay images for each matched name on redaction boxes.
    
    Args:
        matched_pairs: Dictionary mapping box keys (page, line, x, y, w, h) to match info
        pdf_path: Path to PDF file
        glyph_bank: Dictionary of character glyphs
        spacing_model: Optional spacing/kerning model (defaults to simple spacing)
        output_dir: Directory to save overlay images
        dpi: DPI for PDF rendering
        visualize_each: Whether to display each overlay (for notebook usage)
        
    Returns:
        List of created overlay file paths
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    scale = dpi / 96.0
    
    # Create default spacing model if not provided
    if spacing_model is None:
        spacing_model = {
            "page_space_px": 8.0,
            "page_pair_map": {},
            "line_models": []
        }
    
    overlay_files = []
    name_counts = {}  # Track instances of each name
    
    for box_key, match_info in matched_pairs.items():
        page_idx, line_num, x_pt, y_pt, w_pt, h_pt = box_key
        name = match_info["name"]
        
        # Track instance count for this name
        if name not in name_counts:
            name_counts[name] = 0
        name_counts[name] += 1
        instance_num = name_counts[name]
        
        # Render PDF page
        page = doc[page_idx]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Convert box coordinates from points to pixels
        x_px = int(x_pt * scale)
        y_px = int(y_pt * scale)
        w_px = int(w_pt * scale)
        h_px = int(h_pt * scale)
        
        # Extract region around the redaction box (with some padding)
        padding = 50
        y0 = max(0, y_px - padding)
        y1 = min(img.shape[0], y_px + h_px + padding)
        x0 = max(0, x_px - padding)
        x1 = min(img.shape[1], x_px + w_px + padding)
        
        region = img[y0:y1, x0:x1].copy()
        
        # Adjust coordinates relative to extracted region
        box_x_in_region = x_px - x0
        box_y_in_region = y_px - y0
        
        # Render the name using glyphs
        y_center_pt = y_pt + h_pt / 2.0
        glyphs = _render_name_with_glyphs(
            name, glyph_bank, spacing_model,
            y_center_pt, box_x_in_region, box_y_in_region + h_px
        )
        
        # Overlay glyphs onto the region
        for glyph_alpha, gx, gy in glyphs:
            gh, gw = glyph_alpha.shape[:2]
            
            # Clip to region bounds
            gx0 = max(0, gx)
            gy0 = max(0, gy)
            gx1 = min(region.shape[1], gx + gw)
            gy1 = min(region.shape[0], gy + gh)
            
            if gx1 <= gx0 or gy1 <= gy0:
                continue
            
            # Calculate alpha region indices
            ax0 = gx0 - gx
            ay0 = gy0 - gy
            ax1 = ax0 + (gx1 - gx0)
            ay1 = ay0 + (gy1 - gy0)
            
            # Apply alpha blending (green overlay)
            alpha_sub = glyph_alpha[ay0:ay1, ax0:ax1].astype(np.float32) / 255.0
            roi = region[gy0:gy1, gx0:gx1].astype(np.float32)
            
            # Green overlay with 50% opacity
            roi[:, :, 0] *= (1.0 - alpha_sub * 0.5)  # Blue channel
            roi[:, :, 1] = np.clip(roi[:, :, 1] + alpha_sub * 127.5, 0, 255)  # Green channel
            roi[:, :, 2] *= (1.0 - alpha_sub * 0.5)  # Red channel
            
            region[gy0:gy1, gx0:gx1] = np.clip(roi, 0, 255).astype(np.uint8)
        
        # Draw rectangle around redaction box
        cv2.rectangle(
            region,
            (box_x_in_region, box_y_in_region),
            (box_x_in_region + w_px, box_y_in_region + h_px),
            (0, 255, 0), 2
        )
        
        # Save overlay
        safe_name = name.replace(" ", "_").replace("/", "_")
        filename = f"overlay_page{page_idx}_line{line_num}_{safe_name}_instance{instance_num}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, region)
        overlay_files.append(filepath)
        
        print(f"Created overlay for '{name}' (instance {instance_num}) on page {page_idx}, line {line_num}")
        print(f"  Saved to: {filepath}")
        
        # Visualize if requested (for notebook usage)
        if visualize_each:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            plt.title(f"{name} (instance {instance_num}) - Page {page_idx}, Line {line_num}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    doc.close()
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total overlays created: {len(overlay_files)}")
    print(f"Unique names: {len(name_counts)}")
    print(f"Output directory: {output_dir}")
    
    return overlay_files
