"""
Module for extracting specific text from images using various OCR models.

This module provides functions to extract text from specified bounding boxes within images 
using either the LLaVA or DocTR OCR models. The extraction process involves cropping the image 
to the bounding box and then passing it to the appropriate model for text extraction.

Functions:
    parse_bbox(img_path: Path, box: BoundingBox | None, mode: ExtractMode, model: Any, prompt: str) -> str:
        Extracts text from a specified bounding box within an image using the specified model and mode.

    generate_doctr_model() -> OCRPredictor:
        Initializes and returns a pretrained DocTR OCR model.

    parse_bbox_doctr(model: Any, img_path: Path, box: BoundingBox) -> str:
        Extracts text from a specified bounding box within an image using the DocTR model.

    generate_LLaVA_model() -> LLaVA:
        Initializes and returns a pretrained LLaVA model.

    parse_bbox_llava(model: Any, img_path: Path, box: BoundingBox, prompt: str) -> str:
        Extracts text from a specified bounding box within an image using the LLaVA model.

Classes:
    ExtractMode(Enum):
        Enum defining the supported extraction modes (DOC_TR and LLAVA).
"""

from enum import Enum
from pathlib import Path
from extract_bboxes import BoundingBox

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.models.predictor import OCRPredictor
from PIL import Image
from llava_wrapper import LLaVA
from typing import Any
import sys
from extraction_utils import crop_image

TEMPFILE_PATH = "/tmp/tempfile.jpg"

class ExtractMode(Enum):
    DOC_TR = "DocTR"
    LLAVA = "LLaVA"

def parse_bbox(
        img_path: Path,
        box : BoundingBox | None,
        mode : ExtractMode,
        model: Any,
        prompt: str = "Print the text in this image."
) -> str:
    """

    Note that if 'box' is None, the entire image will be considered as the bounding box.

    Ensure that the model matches the mode.
    """

    match mode:
        case ExtractMode.DOC_TR:
            assert isinstance(model, OCRPredictor)
            return parse_bbox_doctr(model, img_path, box)
        
        case ExtractMode.LLAVA:
            assert isinstance(model, LLaVA)
            return parse_bbox_llava(model, img_path, box, prompt)
        
        case _:
            raise ValueError(f"Invalid mode: {mode}")
        

def generate_doctr_model() -> OCRPredictor:
    return (ocr_predictor(pretrained=True)).cuda()

def parse_bbox_doctr(
        model: Any,
        img_path: Path,
        box : BoundingBox,
) -> str:
    """ Parse a bounding box using DocTR model """
    cropped_image = crop_image(img_path, box)
    cropped_image.save(TEMPFILE_PATH)
    doc = DocumentFile.from_images(TEMPFILE_PATH)
    result = model(doc)
    concatString = ""
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                concatString = concatString + word.value + " "
    return concatString[:-1]


def generate_LLaVA_model() -> LLaVA:
    """
    Note that this takes a while to load. 

    If you're intending on using LLaVA to process multiple images,
    it is recommended to only run this once.
    """
    return LLaVA("liuhaotian/llava-v1.6-34b")

def parse_bbox_llava(
        model: Any,
        img_path: Path,
        box : BoundingBox,
        prompt: str
) -> str:
    """ Parse a bounding box using LLAVA model """
    cropped_image = crop_image(img_path, box)
    cropped_image.save(TEMPFILE_PATH)
    return model.eval(TEMPFILE_PATH, prompt)