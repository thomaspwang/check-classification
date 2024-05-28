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

# TODO: Rename 'parse_handwriting' semantic to 'parse_bbox' since we're running it on non-handwriting portions.
# TODO: Get LLaVa working for pip - or make a docker instance?

TEMPFILE_PATH = "/tmp/tempfile.jpg"

class ExtractMode(Enum):
    DOC_TR = "DocTR"
    LLAVA = "LLAVA"

def crop_image(image_path: Path, bbox: BoundingBox):
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    if bbox is None:
        return image
    x1, y1 = bbox.top_left_corner()
    x2, y2 = bbox.bottom_right_corner()
    return image.crop((x1, y1, x2, y2))


def parse_handwriting(
        img_path: Path,
        box : BoundingBox,
        mode : ExtractMode,
        model: Any,
        prompt: str = "Print the text in this image."
) -> str:
    """
    Note: Make sure model matches the mode.
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

# if __name__ == "__main__":
#     if len(sys.argv) != 7:
#         print("Usage: python3 extract_handwriting.py <image_path> <x> <y> <width> <height> <mode>")
#         sys.exit(1)
    
#     # Parse the image path from command-line arguments
#     image_path = Path(sys.argv[1])
#     boundingBox = BoundingBox(x=int(sys.argv[2]), y=int(sys.argv[3]), width=int(sys.argv[4]), height=int(sys.argv[5]))
#     mode = Mode(sys.argv[6])
    
#     parsed_string = parse_handwriting(image_path, boundingBox, mode)
#     print(f"Parsed Handwriting: {parsed_string}")