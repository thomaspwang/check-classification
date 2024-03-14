from enum import Enum
from pathlib import Path
from extract_bboxes import BoundingBox, RegionType

class Mode(Enum):
    DOC_TR = "DocTR"
    TR_OCR = "TrOCR"
    LLAVA = "LLAVA"
    AMAZON_TEXTRACT = "Amazon Textract"
    # add more as needed

def parse_handwriting(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
        mode : Mode,
) -> str:
    match mode:
        case Mode.DOC_TR:
            return parse_handwriting_doctr(img_path, region, box)
        case Mode.TR_OCR:
            return parse_handwriting_trocr(img_path, region, box)
        case Mode.LLAVA:
            return parse_handwriting_llava(img_path, region, box)
        case Mode.AMAZON_TEXTRACT:
            return parse_handwriting_amazon_textract(img_path, region, box)
        case _:
            raise ValueError(f"Invalid mode: {mode}")

def parse_handwriting_doctr(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using DocTR model """
    raise NotImplementedError("This function has not been implemented yet.")

def parse_handwriting_trocr(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using TrOCR model """
    raise NotImplementedError("This function has not been implemented yet.")

def parse_handwriting_llava(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using LLAVA model """
    raise NotImplementedError("This function has not been implemented yet.")

def parse_handwriting_amazon_textract(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using Amazon Textract """
    raise NotImplementedError("This function has not been implemented yet.")