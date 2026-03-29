# Redaction Analysis Package

A unified Python module for PDF redaction box detection, extraction, and analysis. Consolidates utilities previously scattered across the notebook and multiple scripts.

## Features

### Classes

- **`RedactionBox`** - Data class representing a detected redaction box with position and dimensions
- **`PDFSentenceExtractor`** - Detects redaction boxes, extracts text bands, and finds individual character boxes
- **`CharacterOverlay`** - Handles alpha-blended character overlay onto PDF extracts

### Utility Functions

- **`extract_redaction_boxes()`** - Extract all redaction boxes from a PDF with configurable thresholds
- **`get_line_font_info()`** - Extract font name and size for each line in a document
- **`estimate_redaction_char_count()`** - Estimate how many characters could fit in a redaction
- **`analyze_redaction_boxes()`** - Compute statistics about detected boxes

## Installation

```python
# Add to your Python path or install in your virtual environment
import sys
sys.path.insert(0, '/path/to/Pipeline')
from redaction_analysis import PDFSentenceExtractor, extract_redaction_boxes
```

## Quick Start

### Detecting Redaction Boxes

```python
import fitz
from redaction_analysis import extract_redaction_boxes

# Open PDF
doc = fitz.open("document.pdf")

# Extract redaction boxes
boxes = extract_redaction_boxes(
    doc,
    black_threshold=230,
    min_width=30,
    min_height=30,
    visualize=True,
    output_dir="./overlays"
)

# Analyze results
for box in boxes[:5]:
    print(f"\nBox: {box['uuid']}")
    print(f"  Size: {box['width_pix']}×{box['height_pix']} pixels")
    print(f"  Position: ({box['x_pt']:.1f}, {box['y_pt']:.1f}) points")
```

### Extracting Text Regions

```python
import cv2
from redaction_analysis import PDFSentenceExtractor

# Initialize extractor
extractor = PDFSentenceExtractor()

# Detect boxes in PDF image
boxes = extractor.extract_redaction_boxes(pdf_image)

# Extract text band containing boxes
section, box_pos, same_row = extractor.crop_text_band(pdf_image, boxes)

# Find character boxes
char_boxes = extractor.find_character_boxes(section)
print(f"Found {len(char_boxes)} characters")
```

### Overlaying Characters

```python
from PIL import Image
from redaction_analysis import CharacterOverlay

# Create overlay from pre-rendered character
char_image = Image.open("character.png")
overlay = CharacterOverlay(char_image, color=(255, 0, 0), alpha=0.5)

# Find target position by counting from right (position 5 = 5th char from right)
target_pos = overlay.find_target_position(char_boxes, position_from_right=5)

if target_pos:
    x, y, w, h = target_pos
    overlay.blend_onto(pdf_image, (x, y), target_size=(w, h))
```

### Font Analysis

```python
import fitz
from redaction_analysis import get_line_font_info

doc = fitz.open("document.pdf")
font_info = get_line_font_info(doc)

# Display fonts used in first 10 lines
for line_info in font_info[:10]:
    print(f"Page {line_info['page']}, Line {line_info['line_number']}: "
          f"{line_info['font_name']} {line_info['font_size']}pt")
```

## Module Structure

```
redaction_analysis/
├── __init__.py           # Package exports
├── extraction.py         # PDFSentenceExtractor, CharacterOverlay, RedactionBox
└── pdf_utils.py          # Utility functions (extraction, analysis, font info)
```

## API Reference

### `PDFSentenceExtractor`

Main class for PDF text extraction and redaction analysis.

**Methods:**
- `extract_redaction_boxes(image)` - Find solid dark rectangles
- `crop_text_band(image, boxes)` - Extract text region containing boxes
- `find_character_boxes(image)` - Detect individual character positions

### `CharacterOverlay`

Alpha-blended character rendering on PDF images.

**Methods:**
- `find_target_position(char_boxes, position_from_right)` - Locate target character
- `blend_onto(target, position, target_size)` - Apply character overlay

### `extract_redaction_boxes(pdf, ...)`

Detect redaction patterns in a PDF document.

**Parameters:**
- `pdf`: PyMuPDF document object
- `black_threshold`: Darkness threshold (0-255, default: 100)
- `min_width`: Minimum box width in pixels (default: 50)
- `min_height`: Minimum box height in pixels (default: 30)
- `visualize`: Generate visualization images (default: False)
- `output_dir`: Directory for visualizations

**Returns:** List of detected boxes with UUID, position, size, and metadata

### `analyze_redaction_boxes(boxes)`

Compute statistics about detected boxes.

**Returns:** Dict with total count, per-page breakdown, and width/height/area statistics

## Examples

See [examples/](examples/) directory for complete working examples.

## Integration with Existing Code

The package is designed as a drop-in replacement that consolidates existing functionality:

- **`kellen_render.py`** - Now imports classes from the package
- **`Text_inference.ipynb`** - Can now use `from redaction_analysis import extract_redaction_boxes`
- **`analyze_redaction_boxes.py`** - Can be simplified to use package utilities

## License

Part of the Redaction Analysis project.
