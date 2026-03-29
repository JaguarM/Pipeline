"""
Redaction Analysis Package

Unified module for PDF redaction box detection, extraction, and analysis.
Consolidates utilities from Text_inference.ipynb and scattered Python scripts.
"""

from .extraction import (
    RedactionBox,
    PDFSentenceExtractor,
    CharacterOverlay,
)

from .pdf_utils import (
    extract_redaction_boxes,
    get_line_font_info,
    estimate_redaction_char_count,
    analyze_redaction_boxes,
)

__version__ = "0.1.0"
__all__ = [
    "RedactionBox",
    "PDFSentenceExtractor",
    "CharacterOverlay",
    "extract_redaction_boxes",
    "get_line_font_info",
    "estimate_redaction_char_count",
    "analyze_redaction_boxes",
]
