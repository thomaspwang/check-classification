#TODO: Module docstring


from enum import Enum
from pathlib import Path
from extract_bboxes import BoundingBox

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
# from llava_wrapper import LLaVA

import sys

# TODO: Rename 'parse_handwriting' semantic to 'parse_bbox' since we're running it on non-handwriting portions.
# TODO: Get LLaVa working for pip - or make a docker instance?

TEMPFILE_PATH = "/tmp/tempfile.jpg"
MODEL = (ocr_predictor(pretrained=True)).cuda()
# LLAVA = LLaVA("liuhaotian/llava-v1.6-34b")
PROMPT = "print the text in the image."

class Mode(Enum):
    DOC_TR = "DocTR"
    LLAVA = "LLAVA"
    # add more as needed

def crop_image(image_path: Path, bbox: BoundingBox):
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    x1, y1 = bbox.top_left_corner()
    x2, y2 = bbox.bottom_right_corner()
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(TEMPFILE_PATH)


def parse_handwriting(
        img_path: Path,
        box : BoundingBox,
        mode : Mode,
) -> str:
    match mode:
        case Mode.DOC_TR:
            return parse_handwriting_doctr(img_path, box)
        # case Mode.LLAVA:
        #     return parse_handwriting_llava(img_path, box)
        case _:
            raise ValueError(f"Invalid mode: {mode}")


def parse_handwriting_doctr(
        img_path: Path,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using DocTR model """
    crop_image(img_path, box)
    doc = DocumentFile.from_images(TEMPFILE_PATH)
    result = MODEL(doc)
    concatString = ""
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                concatString = concatString + word.value + " "
    return concatString[:-1]


# def parse_handwriting_llava(
#         img_path: Path,
#         box : BoundingBox,
# ) -> str:
#     """ Parse handwriting using LLAVA model """
#     crop_image(img_path, box)
#     return model.eval(TEMPFILE_PATH,PROMPT)

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