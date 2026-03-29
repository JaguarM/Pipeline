
import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from typing import List, Tuple, Optional

# Configuration
PDF_FILE = "./PDFS/vol00008-official-doj-latest-efta00037366.pdf"

def detect_black_bars(pdf_path: str) -> None:
    print(f"Processing {pdf_path} using PyMuPDF...")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    total_bars = 0
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        
        # Render page to image (pixmap)
        pix = page.get_pixmap(dpi=96)
        
        # Convert to numpy array
        # pix.samples is a bytes object
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        
        # Reshape based on height, width, and number of channels (usually 3 for RGB or 4 for RGBA)
        if pix.n >= 3:
            img_array = img_array.reshape(pix.h, pix.w, pix.n)
            # Take only first 3 channels (RGB) if RGBA
            img_array = img_array[:, :, :3]
            # Convert RGB to BGR for OpenCV
            open_cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:
            # Grayscale already? reshape
            img_array = img_array.reshape(pix.h, pix.w)
            open_cv_image = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            print(f"Page {page_num}: Unsupported channel count {pix.n}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find black regions
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bars_on_page = 0
        
        valid_bars = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on dimensions and solidity to ensure it's a bar
            area = cv2.contourArea(contour)
            bbox_area = w * h
            
            if bbox_area > 0:
                solidity = area / float(bbox_area)
            else:
                solidity = 0
            
            # Criteria:
            # 1. Area > 50 to avoid noise
            # 2. Solidity > 0.8 (rectangular)
            if area > 50 and solidity > 0.8:
                valid_bars.append((y, x, w, h, area))

        # Sort bars by Y coordinate (top to bottom), then X (left to right)
        valid_bars.sort(key=lambda b: (b[0], b[1]))

        for y, x, w, h, area in valid_bars:
            print(f"Page {page_num}: Found bar | Width: {w} px | Height: {h} px | Area: {area} px^2")
            bars_on_page += 1
            total_bars += 1

    print(f"\nTotal black bars detected: {total_bars}")
    doc.close()

if __name__ == "__main__":
    if os.path.exists(PDF_FILE):
        detect_black_bars(PDF_FILE)
    else:
        print(f"File not found: {PDF_FILE}")
