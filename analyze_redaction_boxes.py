#!/usr/bin/env python3
"""
Analyze redaction box detection results with threshold=240
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# This script assumes you have 'boxes' variable from the notebook
# Run this in the notebook or load the boxes data

def analyze_boxes(boxes: List[Dict[str, Any]]) -> None:
    """Analyze and display statistics about detected redaction boxes"""
    print(f"Total redaction boxes detected: {len(boxes)}")
    print(f"\nFirst 5 boxes:")
    for i, box in enumerate(boxes[:5]):
        print(f"\nBox {i+1} - UUID: {box['uuid']}")
        print(f"  Page: {box['page_index'] + 1}, Line: {box['line_number']}")
        print(f"  Dimensions: {box['width_pix']}×{box['height_pix']} pixels")
        print(f"  Position: ({box['x_pt']:.1f}, {box['y_pt']:.1f}) points")
    
    # Show width distribution
    widths = [box['width_pix'] for box in boxes]
    heights = [box['height_pix'] for box in boxes]
    print(f"\n\nWidth statistics:")
    print(f"  Min: {min(widths)}px, Max: {max(widths)}px, Mean: {np.mean(widths):.1f}px")
    print(f"Height statistics:")
    print(f"  Min: {min(heights)}px, Max: {max(heights)}px, Mean: {np.mean(heights):.1f}px")
    
    # Display the visualization
    viz_img = Image.open(BASE_DIR / 'overlays' / 'redaction_boxes_page_1.png')
    plt.figure(figsize=(14, 18))
    plt.imshow(viz_img)
    plt.axis('off')
    plt.title(f"Redaction Boxes Detected (threshold=240) - {len([b for b in boxes if b['page_index'] == 0])} boxes on page 1")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Run this code in the notebook after executing the redaction box extraction")
    print("Example:")
    print("  analyze_boxes(boxes)")
