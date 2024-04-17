from enum import Enum
from pathlib import Path
from extract_bboxes import BoundingBox

import sys

class Mode(Enum):
    DOC_TR = "DocTR"
    LLAVA = "LLAVA"
    # add more as needed

def parse_handwriting(
        img_path: Path,
        box : BoundingBox,
        mode : Mode,
) -> str:
    match mode:
        case Mode.DOC_TR:
            return parse_handwriting_doctr(img_path, box)
        case Mode.LLAVA:
            return parse_handwriting_llava(img_path, box)
        case _:
            raise ValueError(f"Invalid mode: {mode}")


def parse_handwriting_doctr(
        img_path: Path,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using DocTR model """
    raise NotImplementedError("This function has not been implemented yet.")

def parse_handwriting_llava(
        img_path: Path,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using LLAVA model """
    raise NotImplementedError("This function has not been implemented yet.")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python3 extract_handwriting.py <image_path> <box_min_x> <box_min_y> <box_max_x> <box_max_y> <mode>")
        sys.exit(1)
    
    # Parse the image path from command-line arguments
    image_path = Path(sys.argv[1])
    boundingBox = BoundingBox(min_x=int(sys.argv[2]), min_y=int(sys.argv[3]), max_x=int(sys.argv[4]), max_y=int(sys.argv[5]))
    mode = Mode(sys.argv[6])
    
    parsed_string = parse_handwriting(image_path, boundingBox, mode)
    print(f"Parsed Handwriting: {parsed_string}")