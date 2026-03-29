"""
Example: Using redaction_analysis package to detect and analyze PDFs

This example demonstrates the main features of the redaction_analysis package:
- Detecting redaction boxes in PDFs
- Extracting text regions
- Analyzing detected patterns
"""

import sys
sys.path.insert(0, '.')

import fitz
from redaction_analysis import (
    extract_redaction_boxes,
    get_line_font_info,
    analyze_redaction_boxes,
    PDFSentenceExtractor,
    CharacterOverlay,
)
import cv2
import numpy as np


def example_1_extract_boxes():
    """Example 1: Extract all redaction boxes from a PDF"""
    print("=" * 70)
    print("Example 1: Extract Redaction Boxes")
    print("=" * 70)
    
    pdf_path = "./PDFS/vol00008-official-doj-latest-efta00037366.pdf"
    
    try:
        doc = fitz.open(pdf_path)
        
        # Extract boxes with custom thresholds
        boxes = extract_redaction_boxes(
            doc,
            black_threshold=230,
            min_width=30,
            min_height=30,
            visualize=True,
            output_dir="./overlays"
        )
        
        print(f"\nFound {len(boxes)} redaction boxes")
        
        # Show first 3 boxes
        for i, box in enumerate(boxes[:3]):
            print(f"\n  Box {i+1}: {box['uuid']}")
            print(f"    Position: ({box['x_pt']:.1f}, {box['y_pt']:.1f}) points")
            print(f"    Size: {box['width_pix']}×{box['height_pix']} pixels")
            print(f"    Area: {box['area_pix']} pixels²")
        
        doc.close()
        
    except FileNotFoundError:
        print("PDF not found. Using example path instead.")


def example_2_analyze_boxes():
    """Example 2: Analyze statistics about detected boxes"""
    print("\n" + "=" * 70)
    print("Example 2: Analyze Box Statistics")
    print("=" * 70)
    
    pdf_path = "./PDFS/vol00008-official-doj-latest-efta00037366.pdf"
    
    try:
        doc = fitz.open(pdf_path)
        boxes = extract_redaction_boxes(doc, black_threshold=230, min_width=30, min_height=30)
        
        # Get statistics
        stats = analyze_redaction_boxes(boxes)
        
        print(f"\nTotal boxes: {stats['total_count']}")
        print(f"Boxes per page: {stats['by_page']}")
        
        print(f"\nWidth statistics (pixels):")
        print(f"  Min: {stats['width_stats']['min']}")
        print(f"  Max: {stats['width_stats']['max']}")
        print(f"  Mean: {stats['width_stats']['mean']:.1f}")
        print(f"  Median: {stats['width_stats']['median']:.1f}")
        
        print(f"\nHeight statistics (pixels):")
        print(f"  Min: {stats['height_stats']['min']}")
        print(f"  Max: {stats['height_stats']['max']}")
        print(f"  Mean: {stats['height_stats']['mean']:.1f}")
        print(f"  Median: {stats['height_stats']['median']:.1f}")
        
        doc.close()
        
    except FileNotFoundError:
        print("PDF not found for analysis.")


def example_3_font_info():
    """Example 3: Extract font information from PDF"""
    print("\n" + "=" * 70)
    print("Example 3: Extract Font Information")
    print("=" * 70)
    
    pdf_path = "./PDFS/vol00008-official-doj-latest-efta00037366.pdf"
    
    try:
        doc = fitz.open(pdf_path)
        font_info = get_line_font_info(doc)
        
        print(f"\nTotal lines analyzed: {len(font_info)}")
        
        # Show first 10 lines
        print("\nFirst 10 lines:")
        for i, line in enumerate(font_info[:10]):
            print(f"  Line {i+1}: {line['font_name']} {line['font_size']}pt (Page {line['page'] + 1})")
        
        # Find unique fonts
        fonts_used = set((f['font_name'], f['font_size']) for f in font_info)
        print(f"\nUnique fonts used: {len(fonts_used)}")
        for font_name, font_size in sorted(fonts_used):
            print(f"  - {font_name} at {font_size}pt")
        
        doc.close()
        
    except FileNotFoundError:
        print("PDF not found for font extraction.")


def example_4_extract_text_band():
    """Example 4: Extract text band containing redaction boxes"""
    print("\n" + "=" * 70)
    print("Example 4: Extract Text Band from PDF")
    print("=" * 70)
    
    pdf_path = "./PDFS/vol00008-official-doj-latest-efta00037366.pdf"
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Render page to image
        mat = fitz.Matrix(300/96, 300/96)
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)
        
        # Extract boxes and text band
        extractor = PDFSentenceExtractor()
        boxes = extractor.extract_redaction_boxes(img)
        section, box_pos, same_row = extractor.crop_text_band(img, boxes)
        
        if section is not None:
            print(f"\nExtracted text band:")
            print(f"  Dimensions: {section.shape[1]}×{section.shape[0]} pixels")
            print(f"  Box position in band: ({box_pos[0]}, {box_pos[1]})")
            print(f"  Box size: {box_pos[2]}×{box_pos[3]} pixels")
            
            # Find character boxes in extracted band
            char_boxes = extractor.find_character_boxes(section)
            print(f"  Found {len(char_boxes)} character boxes")
        else:
            print("No text band extracted (boxes not found)")
        
        doc.close()
        
    except Exception as e:
        print(f"Error: {e}")


def example_5_character_overlay():
    """Example 5: Use CharacterOverlay for character positioning"""
    print("\n" + "=" * 70)
    print("Example 5: Character Overlay Positioning")
    print("=" * 70)
    
    pdf_path = "./PDFS/vol00008-official-doj-latest-efta00037366.pdf"
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test character image
        char_img = Image.new('RGB', (20, 30), color=(255, 255, 255))
        draw = ImageDraw.Draw(char_img)
        draw.rectangle([2, 2, 18, 28], outline=(255, 0, 0), width=2)
        
        # Create overlay
        overlay = CharacterOverlay(char_img, color=(255, 0, 0), alpha=0.5)
        
        print(f"\nCharacterOverlay created:")
        print(f"  Reference image shape: {overlay.ref_img.shape}")
        print(f"  Cropped image shape: {overlay.cropped_img.shape}")
        print(f"  Alpha transparency: {overlay.alpha}")
        
        # Example character boxes (x, y, w, h)
        char_boxes = [
            (10, 50, 15, 25),   # Character 1
            (30, 50, 14, 26),   # Character 2
            (50, 50, 16, 24),   # Character 3
            (70, 50, 15, 25),   # Character 4
        ]
        
        # Find 2nd character from right
        target = overlay.find_target_position(char_boxes, position_from_right=2)
        
        print(f"\n  Looking for 2nd character from right")
        print(f"  Found at position: {target}")
        
    except ImportError:
        print("PIL not available for this example")


if __name__ == "__main__":
    print("\nRedaction Analysis Package - Examples\n")
    
    # Run examples
    example_1_extract_boxes()
    example_2_analyze_boxes()
    example_3_font_info()
    example_4_extract_text_band()
    example_5_character_overlay()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
