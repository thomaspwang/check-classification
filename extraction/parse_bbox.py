#TODO: Module docstring


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

# TODO: Rename 'parse_handwriting' semantic to 'parse_bbox' since we're running it on non-handwriting portions.
# TODO: Get LLaVa working for pip - or make a docker instance?

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
            return parse_handwriting_doctr(model, img_path, box)
        
        case ExtractMode.LLAVA:
            assert isinstance(model, LLaVA)
            return parse_handwriting_llava(model, img_path, box, prompt)
        
        case _:
            raise ValueError(f"Invalid mode: {mode}")
        

def generate_doctr_model() -> OCRPredictor:
    return (ocr_predictor(pretrained=True)).cuda()

def parse_handwriting_doctr(
        model: Any,
        img_path: Path,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using DocTR model """
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

def parse_handwriting_llava(
        model: Any,
        img_path: Path,
        box : BoundingBox,
        prompt: str
) -> str:
    """ Parse handwriting using LLAVA model """
    cropped_image = crop_image(img_path, box)
    cropped_image.save(TEMPFILE_PATH)
    return model.eval(TEMPFILE_PATH, prompt)