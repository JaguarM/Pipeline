#!/usr/bin/env python3
"""
Extract character-level bounding boxes from PDF using PyMuPDF.
Shows the exact bbox for each character, as you would see when selecting text.
"""

import fitz
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

def extract_char_bboxes(pdf_path: str, page_num: int = 0, output_path: str = "char_bboxes_visualization.png") -> List[Dict[str, Any]]:
    """
    Extract character-level bounding boxes from a PDF page.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        output_path: Where to save the visualization
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Get page dimensions
    page_rect = page.rect
    print(f"Page size: {page_rect.width}x{page_rect.height} points")
    
    # Render page as image at high resolution (3.125x scale matches PDF extraction)
    zoom = 3.125
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Get character-level text information (use "rawdict" for char-level bboxes)
    text_dict = page.get_text("rawdict")
    
    print(f"\nExtracting character bboxes from page {page_num + 1}...")
    
    char_count = 0
    char_data = []
    
    # Iterate through all text blocks, lines, spans, and characters
    for block_idx, block in enumerate(text_dict["blocks"]):
        if block["type"] == 0:  # Text block (type 0), ignore image blocks (type 1)
            for line_idx, line in enumerate(block["lines"]):
                for span_idx, span in enumerate(line["spans"]):
                    font = span.get("font", "Unknown")
                    size = span.get("size", 0)
                    
                    for char_idx, char in enumerate(span["chars"]):
                        c = char["c"]
                        bbox = char["bbox"]  # (x0, y0, x1, y1) in PDF points
                        origin = char.get("origin", (0, 0))
                        
                        char_data.append({
                            "char": c,
                            "bbox": bbox,
                            "origin": origin,
                            "font": font,
                            "size": size,
                            "block": block_idx,
                            "line": line_idx,
                            "span": span_idx
                        })
                        
                        char_count += 1
                        
                        # Draw bbox on image (scale from PDF points to image pixels)
                        x0, y0, x1, y1 = bbox
                        x0_px = int(x0 * zoom)
                        y0_px = int(y0 * zoom)
                        x1_px = int(x1 * zoom)
                        y1_px = int(y1 * zoom)
                        
                        # Draw rectangle (thin blue line)
                        cv2.rectangle(img, (x0_px, y0_px), (x1_px, y1_px), (255, 0, 0), 1)
    
    print(f"Found {char_count} characters")
    
    # Save visualization
    cv2.imwrite(output_path, img)
    print(f"✓ Saved visualization to {output_path}")
    
    # Print first 20 characters as examples
    print(f"\nFirst 20 characters:")
    for i, cd in enumerate(char_data[:20]):
        print(f"  [{i}] '{cd['char']}' bbox: ({cd['bbox'][0]:.1f}, {cd['bbox'][1]:.1f}, {cd['bbox'][2]:.1f}, {cd['bbox'][3]:.1f}) "
              f"font: {cd['font']} {cd['size']:.1f}pt")
    
    doc.close()
    
    return char_data


def export_char_bboxes_to_text(char_data: List[Dict[str, Any]], output_path: str = "char_bboxes_data.txt") -> None:
    """
    Export character bbox data to a text file for inspection.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Total characters: {len(char_data)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, cd in enumerate(char_data):
            f.write(f"[{i:4d}] char: '{cd['char']}'\n")
            f.write(f"       bbox: ({cd['bbox'][0]:7.2f}, {cd['bbox'][1]:7.2f}, {cd['bbox'][2]:7.2f}, {cd['bbox'][3]:7.2f})\n")
            f.write(f"       origin: ({cd['origin'][0]:7.2f}, {cd['origin'][1]:7.2f})\n")
            f.write(f"       font: {cd['font']}, size: {cd['size']:.1f}pt\n")
            f.write(f"       block: {cd['block']}, line: {cd['line']}, span: {cd['span']}\n")
            f.write("\n")
    
    print(f"✓ Exported character data to {output_path}")


def find_text_chars(char_data: List[Dict[str, Any]], search_text: str) -> List[Dict[str, Any]]:
    """
    Find characters matching a specific text string.
    """
    results = []
    search_len = len(search_text)
    
    for i in range(len(char_data) - search_len + 1):
        # Check if next N characters match search text
        match = True
        for j in range(search_len):
            if char_data[i + j]["char"] != search_text[j]:
                match = False
                break
        
        if match:
            # Found a match
            results.append({
                "start_idx": i,
                "end_idx": i + search_len - 1,
                "chars": char_data[i:i + search_len]
            })
    
    return results


if __name__ == "__main__":
    # Path to the PDF we've been working with
    pdf_path = str(BASE_DIR / "PDFS" / "vol00008-official-doj-latest-efta00037366.pdf")
    
    # Extract character bboxes
    char_data = extract_char_bboxes(pdf_path, page_num=0, output_path="char_bboxes_visualization.png")
    
    # Export to text file
    export_char_bboxes_to_text(char_data, output_path="char_bboxes_data.txt")
    
    # Find specific text example: "Contact was made"
    print("\n" + "=" * 80)
    print("Searching for 'Contact was made'...")
    matches = find_text_chars(char_data, "Contact was made")
    
    if matches:
        print(f"Found {len(matches)} match(es):")
        for match_idx, match in enumerate(matches):
            print(f"\n  Match {match_idx + 1}:")
            print(f"  Character indices: {match['start_idx']} to {match['end_idx']}")
            print(f"  Bounding box span: ({match['chars'][0]['bbox'][0]:.1f}, {match['chars'][0]['bbox'][1]:.1f}) to "
                  f"({match['chars'][-1]['bbox'][2]:.1f}, {match['chars'][-1]['bbox'][3]:.1f})")
            print(f"  Characters:")
            for i, cd in enumerate(match['chars']):
                print(f"    [{i}] '{cd['char']}' @ ({cd['bbox'][0]:.1f}, {cd['bbox'][1]:.1f}, {cd['bbox'][2]:.1f}, {cd['bbox'][3]:.1f})")
    else:
        print("  No matches found")
    
    print("\n" + "=" * 80)
    print("Done!")
