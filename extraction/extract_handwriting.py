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
    raise NotImplementedError("This function has not been implemented yet.")