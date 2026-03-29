"""
PDF extraction and character overlay classes for redaction analysis.

This module provides core classes for detecting redaction boxes, extracting text bands,
and overlaying characters on PDF documents.
"""

import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy.typing as npt


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

# Colors
COLOR_RED = (255, 0, 0)
COLOR_OVERLAY_ALPHA = 0.5


# ============================================================================
# DATACLASS
# ============================================================================

@dataclass
class RedactionBox:
    """Represents a detected redaction box in a PDF image.
    
    Attributes:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        w: Width in pixels
        h: Height in pixels
        fill_ratio: Ratio of filled pixels (0.0-1.0)
    """
    x: int
    y: int
    w: int
    h: int
    fill_ratio: float


# ============================================================================
# PDF SENTENCE EXTRACTOR
# ============================================================================

class PDFSentenceExtractor:
    """Extracts and processes text regions from PDF renders.
    
    This class handles detection of redaction boxes, extraction of text bands,
    and character-level analysis for overlay positioning.
    
    Example:
        >>> extractor = PDFSentenceExtractor()
        >>> boxes = extractor.extract_redaction_boxes(pdf_image)
        >>> section, pos, boxes = extractor.crop_text_band(pdf_image, boxes)
        >>> char_boxes = extractor.find_character_boxes(section)
    """
    
    def __init__(
        self,
        section_bounds: Tuple[int, int, int, int] = (PDF_SECTION_Y0, PDF_SECTION_Y1, PDF_SECTION_X0, PDF_SECTION_X1),
        min_box_size: Tuple[int, int] = (MIN_BOX_WIDTH, MIN_BOX_HEIGHT),
        fill_threshold: float = FILL_RATIO_THRESHOLD,
    ):
        """Initialize PDF text extractor.
        
        Args:
            section_bounds: (y0, y1, x0, x1) bounds for text extraction region
            min_box_size: (min_width, min_height) for detecting redaction boxes
            fill_threshold: Minimum fill ratio to classify as redaction box
        """
        self.sec_y0, self.sec_y1, self.sec_x0, self.sec_x1 = section_bounds
        self.min_box_w, self.min_box_h = min_box_size
        self.fill_threshold = fill_threshold
    
    def extract_redaction_boxes(self, image: npt.NDArray[np.uint8]) -> List[RedactionBox]:
        """Detect solid dark rectangles (redaction boxes) in image.
        
        Args:
            image: PDF image region to analyze
            
        Returns:
            List of detected RedactionBox objects, sorted by (y, x) position
        """
        section_region = image[self.sec_y0:self.sec_y1, self.sec_x0:self.sec_x1]
        
        gray_sec = cv2.cvtColor(section_region, cv2.COLOR_BGR2GRAY)
        _, bin_sec = cv2.threshold(gray_sec, 200, 255, cv2.THRESH_BINARY_INV)
        contours_sec, _ = cv2.findContours(bin_sec, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours_sec:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < self.min_box_w or h < self.min_box_h:
                continue
            area = w * h
            if area <= 0:
                continue
            fill_ratio = float(np.count_nonzero(bin_sec[y:y+h, x:x+w])) / float(area)
            if fill_ratio >= self.fill_threshold:
                boxes.append(RedactionBox(x, y, w, h, fill_ratio))
        
        # Sort by (y, x) position
        boxes.sort(key=lambda b: (b.y, b.x))
        return boxes
    
    def crop_text_band(
        self,
        image: npt.NDArray[np.uint8],
        boxes: List[RedactionBox],
    ) -> Tuple[Optional[npt.NDArray[np.uint8]], Optional[Tuple[int, int, int, int]], List[RedactionBox]]:
        """Extract text band containing redaction boxes.
        
        Args:
            image: Full PDF image
            boxes: List of detected redaction boxes
            
        Returns:
            Tuple of (extracted_section, box_positions, same_row_boxes)
            Returns (None, None, []) if no valid boxes found
        """
        if not boxes:
            return None, None, []
        
        section_region = image[self.sec_y0:self.sec_y1, self.sec_x0:self.sec_x1]
        gray_sec = cv2.cvtColor(section_region, cv2.COLOR_BGR2GRAY)
        _, bin_sec = cv2.threshold(gray_sec, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Use BOTTOMMOST row (highest Y value) to get "Contact was made with" not earlier text
        row_y_ref = max(b.y for b in boxes)
        same_row = [b for b in boxes if abs(b.y - row_y_ref) <= 8]
        same_row.sort(key=lambda b: b.x)
        
        if len(same_row) >= 2:
            first_box = same_row[0]
            second_box = same_row[1]
        elif len(same_row) == 1:
            first_box = same_row[0]
            # Create dummy second_box using right edge
            second_box = RedactionBox(first_box.x + first_box.w - 10, first_box.y, 10, first_box.h, first_box.fill_ratio)
        else:
            first_box = boxes[-1]
            second_box = boxes[-1]
        
        # Extract band containing the boxes
        band_y0 = max(0, min(first_box.y, second_box.y) - 2)
        band_y1 = min(section_region.shape[0], max(first_box.y + first_box.h, second_box.y + second_box.h) + 2)
        
        # Find leftmost ink, crop through end of second box
        row_band = bin_sec[band_y0:band_y1, :]
        cols_nonwhite = np.any(row_band > 0, axis=0)
        if cols_nonwhite.any():
            left_col = int(np.where(cols_nonwhite)[0].min())
        else:
            left_col = max(0, first_box.x - 250)
        right_col = min(section_region.shape[1], second_box.x + second_box.w)
        
        extracted = section_region[band_y0:band_y1, left_col:right_col]
        
        # Adjust box positions relative to extracted region
        box_positions = (
            max(0, first_box.x - left_col),
            max(0, first_box.y - band_y0),
            first_box.w,
            first_box.h
        )
        
        return extracted, box_positions, same_row
    
    def find_character_boxes(
        self,
        image: npt.NDArray[np.uint8],
        min_size: Tuple[int, int] = (CHAR_MIN_WIDTH, CHAR_MIN_HEIGHT),
    ) -> List[Tuple[int, int, int, int]]:
        """Find individual character bounding boxes in image.
        
        Args:
            image: Grayscale or color image containing text
            min_size: (min_width, min_height) to filter noise
            
        Returns:
            List of (x, y, w, h) tuples sorted left-to-right
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > min_size[0] and h > min_size[1]:
                boxes.append((x, y, w, h))
        
        boxes.sort(key=lambda b: b[0])  # Sort left to right
        return boxes


# ============================================================================
# CHARACTER OVERLAY
# ============================================================================

class CharacterOverlay:
    """Handles overlay of pre-rendered characters onto PDF extracts.
    
    This class manages character image preparation, positioning, and alpha-blended
    rendering for character-level replacements.
    
    Example:
        >>> overlay = CharacterOverlay(char_image, color=(255, 0, 0), alpha=0.5)
        >>> target_pos = overlay.find_target_position(char_boxes, position_from_right=5)
        >>> overlay.blend_onto(pdf_image, (x, y), target_size=(w, h))
    """
    
    def __init__(
        self,
        reference_image: Image.Image,
        color: Tuple[int, int, int] = COLOR_RED,
        alpha: float = COLOR_OVERLAY_ALPHA,
    ):
        """Initialize character overlay handler.
        
        Args:
            reference_image: Pre-rendered character image (PIL Image)
            color: RGB color tuple for the character
            alpha: Transparency level (0.0 = transparent, 1.0 = opaque)
        """
        self.ref_img = np.array(reference_image)
        self.color = color
        self.alpha = alpha
        self._prepare_alpha_mask()
    
    def _prepare_alpha_mask(self) -> None:
        """Prepare alpha mask from reference image by detecting non-white pixels."""
        # Check for any pixel that's not pure white
        self.mask = np.any(self.ref_img != [255, 255, 255], axis=2).astype(np.uint8) * 255
        
        # Tight crop to character bounds
        nz = cv2.findNonZero(self.mask)
        if nz is not None:
            x, y, w, h = cv2.boundingRect(nz)
            self.cropped_img = self.ref_img[y:y+h, x:x+w]
            self.cropped_mask = self.mask[y:y+h, x:x+w]
        else:
            self.cropped_img = self.ref_img
            self.cropped_mask = self.mask
    
    def find_target_position(
        self,
        char_boxes: List[Tuple[int, int, int, int]],
        position_from_right: int,
        fallback_x: int = 0,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find target character position by counting from right.
        
        Args:
            char_boxes: List of (x, y, w, h) character bounding boxes
            position_from_right: Character position counting from right (1-indexed)
            fallback_x: X position to use as fallback if not found
            
        Returns:
            (x, y, w, h) tuple for target character, or None if not found
        """
        if len(char_boxes) < position_from_right:
            return None
        
        target_index = len(char_boxes) - position_from_right
        if 0 <= target_index < len(char_boxes):
            return char_boxes[target_index]
        return None
    
    def blend_onto(
        self,
        target: npt.NDArray[np.uint8],
        position: Tuple[int, int],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> npt.NDArray[np.uint8]:
        """Blend character onto target image with alpha transparency.
        
        Args:
            target: Target image to blend onto (modified in-place)
            position: (x, y) position for top-left of character
            target_size: Optional (width, height) to resize character before blending
            
        Returns:
            Modified target image (same as input, modified in-place)
        """
        img = self.cropped_img.copy()
        mask = self.cropped_mask.copy()
        
        # Resize if target size specified
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
        
        overlay_h, overlay_w = img.shape[:2]
        place_x, place_y = position
        
        # Clip to target bounds
        x0 = max(0, place_x)
        y0 = max(0, place_y)
        x1 = min(target.shape[1], place_x + overlay_w)
        y1 = min(target.shape[0], place_y + overlay_h)
        
        if x1 <= x0 or y1 <= y0:
            return target
        
        # Calculate alpha region indices
        ax0 = x0 - place_x
        ay0 = y0 - place_y
        ax1 = ax0 + (x1 - x0)
        ay1 = ay0 + (y1 - y0)
        
        # Apply alpha blending
        alpha_sub = mask[ay0:ay1, ax0:ax1].astype(np.float32) / 255.0 * self.alpha
        img_sub = img[ay0:ay1, ax0:ax1].astype(np.float32)
        region = target[y0:y1, x0:x1].astype(np.float32)
        
        for c in range(3):
            region[:, :, c] = region[:, :, c] * (1.0 - alpha_sub) + img_sub[:, :, c] * alpha_sub
        
        target[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
        return target
