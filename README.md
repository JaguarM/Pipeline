# Pipeline

Utilities and experiments for analyzing PDF redactions, extracting character bounding boxes, building glyph assets, and rendering redaction overlays.

## Overview

This repository contains a small redaction-analysis workflow built around PyMuPDF, PIL, and OpenCV-style image processing. It includes:

- PDF redaction box detection
- Character bounding-box extraction
- Glyph bank generation and atlas rendering
- Overlay rendering for candidate text reconstructions
- Helper scripts for page-level visualization and analysis

The reusable package lives in [redaction_analysis/README.md](redaction_analysis/README.md).

## Main files

- [reveal_redacted_chars.py](reveal_redacted_chars.py) — renders recovered character overlays into output images
- [extract_char_bboxes.py](extract_char_bboxes.py) — extracts and visualizes character bounding boxes from a PDF
- [analyze_redaction_boxes.py](analyze_redaction_boxes.py) — analyzes detected redaction boxes and visual output
- [overlay_function.py](overlay_function.py) — overlay rendering helpers
- [kellen_render.py](kellen_render.py) — rendering and comparison utilities
- [contact_line_ms_overlay.py](contact_line_ms_overlay.py) — targeted contact-line overlay generation
- [redaction_analysis](redaction_analysis) — reusable package for extraction and PDF utilities
- [Text_inference.ipynb](Text_inference.ipynb) — notebook workflow and experiments

## Setup

Use a Python virtual environment and install the dependencies used by the scripts.

Typical packages used in this repo include:

- `pymupdf`
- `pillow`
- `numpy`
- `matplotlib`
- `opencv-python`

If your environment is already active, you can run the scripts directly from the repo root.

## Example usage

Run an analysis script from the repository root:

```bash
python analyze_redaction_boxes.py
python extract_char_bboxes.py
python reveal_redacted_chars.py
```

For package-style usage examples, see [redaction_analysis/examples/usage_examples.py](redaction_analysis/examples/usage_examples.py).

## Repository layout

### Source inputs

- [PDFS](PDFS) — source PDFs
- [fonts](fonts) — font files used for rendering/matching
- [glyph_bank](glyph_bank) — glyph image assets

### Generated outputs

These are pipeline outputs and are intentionally ignored by Git:

- `char_bbox_outputs/`
- `redacted_char_outputs/`
- `overlays/`
- generated top-level PNG visualizations
- Python cache and virtual environment files

## Notes

- Large binary inputs such as PDFs, fonts, and spreadsheets are currently tracked in the repo.
- Generated outputs are kept locally but excluded from future commits via [.gitignore](.gitignore).
- The package-level documentation with API details is in [redaction_analysis/README.md](redaction_analysis/README.md).
