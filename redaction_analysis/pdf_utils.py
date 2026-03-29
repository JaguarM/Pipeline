"""
PDF utility functions for redaction analysis.

This module provides utility functions extracted from Text_inference.ipynb,
including redaction box detection, font analysis, and character counting.
"""

import numpy as np
import cv2
import fitz
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


# ============================================================================
# REDACTION BOX EXTRACTION
# ============================================================================

def extract_redaction_boxes(
    pdf: Any,
    black_threshold: int = 100,
    min_width: int = 50,
    min_height: int = 30,
    min_aspect_ratio: float = 1.5,
    visualize: bool = False,
    output_dir: Optional[str] = None,
    dpi: int = 300,
) -> List[Dict[str, Any]]:
    """Extract all types of redaction boxes from a PDF document.
    
    Detects solid black rectangles and other redaction patterns using configurable
    thresholds for size, aspect ratio, and darkness. Optionally generates visualizations.
    
    Args:
        pdf: PyMuPDF document object (fitz.Document)
        black_threshold: Threshold for considering pixels as black (0-255)
        min_width: Minimum width in pixels for a valid redaction box
        min_height: Minimum height in pixels for a valid redaction box
        min_aspect_ratio: Minimum width/height ratio to consider as redaction
        visualize: Whether to create visualization images
        output_dir: Directory to save visualizations (if visualize=True)
        dpi: Dots per inch for PDF rendering
        
    Returns:
        List of detected boxes, each with:
            - uuid: Unique identifier for the box
            - page_index: Zero-indexed page number
            - line_number: Detected line number
            - x_pt, y_pt: Position in points
            - width_pix, width_pt: Width in pixels and points
            - height_pix: Height in pixels
            - area_pix: Area in pixels squared
    """
    boxes = []
    scale = dpi / 96.0
    
    # Extract document number from first page
    doc_number = None
    first_page = pdf[0]
    page_text = first_page.get_text()
    for line in page_text.split('\n'):
        line = line.strip()
        if 'EFTA' in line:
            import re
            match = re.search(r'EFTA\d+', line)
            if match:
                doc_number = match.group(0)
                break
    
    if doc_number is None:
        doc_number = "UNKNOWN"
    
    boxes_per_page = {}
    
    for page_index in range(len(pdf)):
        page = pdf[page_index]
        boxes_per_page[page_index] = 0
        
        # Render page to image
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        
        if pix.n >= 3:
            img_array = img_array.reshape(pix.h, pix.w, pix.n)
            img_array = img_array[:, :, :3]
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:
            img_array = img_array.reshape(pix.h, pix.w)
            img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold and dilate
        _, raw_thresh = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((1, 9), np.uint8)
        thresh = cv2.dilate(raw_thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for idx, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter by size
            if w < min_width or h < min_height:
                continue
            
            # Filter by aspect ratio
            aspect_ratio = w / float(max(1, h))
            if aspect_ratio < min_aspect_ratio:
                continue
            
            # Convert to points
            x_pt = x / scale
            y_pt = y / scale
            w_pt = w / scale
            h_pt = h / scale
            
            # Create unique identifier
            uuid = f"{doc_number}-{page_index + 1}-{len(boxes_per_page[page_index]) + 1}"
            
            box = {
                'uuid': uuid,
                'page_index': page_index,
                'line_number': page_index + 1,  # Simplified line detection
                'x_pix': x,
                'y_pix': y,
                'width_pix': w,
                'height_pix': h,
                'x_pt': x_pt,
                'y_pt': y_pt,
                'width_pt': w_pt,
                'height_pt': h_pt,
                'area_pix': w * h,
                'aspect_ratio': aspect_ratio,
            }
            
            boxes.append(box)
            boxes_per_page[page_index] += 1
        
        # Generate visualization if requested
        if visualize and output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Draw boxes on image
            viz_img = img.copy()
            for box in [b for b in boxes if b['page_index'] == page_index]:
                x, y, w, h = box['x_pix'], box['y_pix'], box['width_pix'], box['height_pix']
                cv2.rectangle(viz_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(viz_img, box['uuid'], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save visualization
            output_path = os.path.join(output_dir, f"redaction_boxes_page_{page_index + 1}.png")
            cv2.imwrite(output_path, viz_img)
            print(f"Saved visualization: {output_path}")
    
    return boxes


# ============================================================================
# FONT INFO EXTRACTION
# ============================================================================

def get_line_font_info(pdf: Any) -> List[Dict[str, Any]]:
    """Extract font information for each line in a PDF document.
    
    Analyzes text structure to determine dominant font name and size for each line.
    Returns a list with font metadata per line.
    
    Args:
        pdf: PyMuPDF document object (fitz.Document)
        
    Returns:
        List of dicts with keys:
            - page: Zero-indexed page number
            - line_number: Line index on the page
            - font_size: Font size in points
            - font_name: Font name string
    """
    results = []
    
    for page_index, page in enumerate(pdf):
        text_dict = page.get_text("dict")
        line_number = 0
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                line_number += 1
                spans = line.get("spans", [])
                
                if not spans:
                    continue
                
                # Extract font pairs (size, name) and find most common
                font_pairs = [
                    (round(span.get("size", 12), 2), span.get("font", "unknown"))
                    for span in spans
                ]
                
                most_common = Counter(font_pairs).most_common(1)
                if most_common:
                    font_size, font_name = most_common[0][0]
                else:
                    font_size, font_name = 12, "unknown"
                
                results.append({
                    "page": page_index,
                    "line_number": line_number,
                    "font_size": font_size,
                    "font_name": font_name,
                })
    
    return results


# ============================================================================
# CHARACTER COUNTING
# ============================================================================

def estimate_redaction_char_count(
    redaction_width: int,
    min_char_width: int,
    max_char_width: int,
) -> Dict[str, Any]:
    """Estimate the character count that could fit in a redaction box.
    
    Uses character width metrics to determine bounds on how many characters
    could be hidden under a redaction, useful for name matching.
    
    Args:
        redaction_width: Width of the redaction box in pixels
        min_char_width: Minimum character width in pixels (narrowest letter like 'i')
        max_char_width: Maximum character width in pixels (widest letter like 'W')
        
    Returns:
        Dict with analysis:
            - min_chars: Minimum characters (using widest char width)
            - max_chars: Maximum characters (using narrowest char width)
            - range: Tuple of (min_chars, max_chars)
            - redaction_width: Original input width
            - min_char_width: Original input min width
            - max_char_width: Original input max width
    """
    # Minimum characters: if all chars are the widest
    min_chars = int(redaction_width / max_char_width)
    
    # Maximum characters: if all chars are the narrowest
    max_chars = int(redaction_width / min_char_width)
    
    return {
        "redaction_width": redaction_width,
        "min_char_width": min_char_width,
        "max_char_width": max_char_width,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "range": (min_chars, max_chars),
    }


# ============================================================================
# ANALYSIS & STATISTICS
# ============================================================================

def analyze_redaction_boxes(boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze and compute statistics about detected redaction boxes.
    
    Args:
        boxes: List of detected boxes from extract_redaction_boxes()
        
    Returns:
        Dict with statistics:
            - total_count: Total number of boxes
            - by_page: Count of boxes per page
            - width_stats: Min/max/mean width in pixels
            - height_stats: Min/max/mean height in pixels
            - area_stats: Min/max/mean area in pixels squared
    """
    if not boxes:
        return {"total_count": 0, "by_page": {}, "boxes_empty": True}
    
    widths = [b['width_pix'] for b in boxes]
    heights = [b['height_pix'] for b in boxes]
    areas = [b['area_pix'] for b in boxes]
    
    by_page = {}
    for box in boxes:
        page = box['page_index']
        by_page[page] = by_page.get(page, 0) + 1
    
    return {
        "total_count": len(boxes),
        "by_page": by_page,
        "width_stats": {
            "min": min(widths),
            "max": max(widths),
            "mean": float(np.mean(widths)),
            "median": float(np.median(widths)),
        },
        "height_stats": {
            "min": min(heights),
            "max": max(heights),
            "mean": float(np.mean(heights)),
            "median": float(np.median(heights)),
        },
        "area_stats": {
            "min": min(areas),
            "max": max(areas),
            "mean": float(np.mean(areas)),
            "median": float(np.median(areas)),
        },
    }
