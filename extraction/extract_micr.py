"""
Module for extracting MICR (Magnetic Ink Character Recognition) data from check images using AWS Textract.

This script processes a check image, extracts bounding boxes using AWS Textract, merges the relevant bounding boxes, 
and then extracts the MICR data (routing number, account number, and check number) from the merged bounding box.

Usage:
    python extract_micr.py <image_path>

Example:
    python extract_micr.py data/mcd-test-3-front-images/mcd-test-3-front-93.jpg
"""

import boto3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from PIL import Image
import re
import tempfile
import sys
from extract_bboxes import (
    extract_bounding_boxes_from_path,
    draw_bounding_boxes_on_image,
    merge_overlapping_boxes,
    BoundingBox
)
from extraction_utils import (
    crop_image,
    merge_n_bounding_boxes,
    largest_area_bounding_box,
    stretch_bounding_box
)

AWS_PROFILE_NAME = 'thwang'
AWS_REGION_NAME = 'us-west-2'

# The amount removed from the top, aka (1 - CROPPING_PERCENTAGE) from the bottom.
# This number was calculated from taking the maximum MICR bbox ratio of all MICR check samples.
# TODO: This is a pretty naive solution, would probably be better to calculate from the bottom bottom and up.
CROPPING_PERCENTAGE = 0.80

@dataclass(frozen=True)
class MICRData:
    """ Represents MICR data.
    
    Note that __eq__ and __hash__ are set implicity by @dataclass.
    """
    routing_number: str
    account_number: str
    check_number: str


class MICRExtractionError(Exception):
    """Exception raised for errors in the MICR format.
    
    This is most likely caused by Textract mistakes.
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def extract_micr_data(
        image_path: Path,
        textract_client,
) -> MICRData | None:
    """ Extract MICR data from check image
    
    Args:
        image_path (Path): Path to check image
        textract_client: An AWS boto3 textract_client.
    
    Returns:
        MICRInfo: Class with the three fields contained in the MICR data
        None: Returns None if cannot parse MICR data.
        
    Note that there are two crops that occur here:
        1. `lateral_crop_file` crops everything but the lateral section bounded by the MICR.
        2. `bbox_crop_file` has everything but the MICR bbox generated by textract cropped out
    """
    # image = Image.open(image_path)
    # width, height = image.size
    # top = int(height * CROPPING_PERCENTAGE)
    # bbox = BoundingBox(0, top, width, height - top)
    # cropped_image = crop_image(image_path, bbox)
    image = Image.open(image_path)
    width, _ = image.size

    first_micr_bbox: BoundingBox = find_first_micr_bounding_box(image_path, textract_client)

    if first_micr_bbox is None:
        raise MICRExtractionError("No MICR bounding box containing '⑆' was identified.")
    
    # We want to crop the image only include the y-coordinate span of 'first_micr_bbox',
    # since it is mostly guaranteed the rest of the MICR is on this lateral. This assumes
    # the check is near-perfectly horizontal.
    micr_bbox_span = first_micr_bbox
    micr_bbox_span.x = 0
    micr_bbox_span.width  = width

    cropped_image = crop_image(image_path, micr_bbox_span)

    bounding_boxes: list[BoundingBox]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as lateral_crop_file:
        lateral_crop_path = Path(lateral_crop_file.name)
        cropped_image.save(lateral_crop_path)
        bounding_boxes = extract_bounding_boxes_from_path(lateral_crop_path, textract_client)

        if len(bounding_boxes) == 0:
            raise MICRExtractionError("No MICR bounding box was identified.")
        
        draw_bounding_boxes_on_image(lateral_crop_path, bounding_boxes, "testing.jpg")
        
        micr_bounding_box = largest_area_bounding_box(bounding_boxes)
        micr_bounding_box = stretch_bounding_box(micr_bounding_box, percentage=0.03)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as bbox_crop_file:
            bbox_crop_path = Path(bbox_crop_file.name)
            image: Image = crop_image(lateral_crop_path, micr_bounding_box)

            # TODO: It would probably be more efficient to create a BytesIO object sending image data to Textract
            #       instead of using a tempfile. But it works for now.
            image.save(bbox_crop_path)

            with bbox_crop_path.open(mode="rb") as f:
                response = textract_client.detect_document_text(
                    Document={'Bytes': f.read()}
                    )

    blocks = response['Blocks']
    micr_string: str = ""
        
    for block in blocks:
        if block['BlockType'] == 'LINE':
            micr_string = micr_string + block['Text']

    micr_data: MICRData = parse_micr_string(micr_string)
    return micr_data

def find_first_micr_bounding_box(
        image_path: Path,
        textract_client,
) -> BoundingBox | None:
    """ Finds the first bounding box that contains the MICR symbol '⑆'. Returns None otherwise.
    """
    with image_path.open(mode="rb") as f:
        response = textract_client.detect_document_text(
            Document={'Bytes': f.read()}
        )
    
    image = Image.open(image_path)
    width, height = image.size
    blocks = response['Blocks']

    for block in blocks:
        if block['BlockType'] == 'LINE' and '⑆' in block['Text']:
            x = int(block['Geometry']['BoundingBox']['Left'] * width)
            y = int(block['Geometry']['BoundingBox']['Top'] * height)
            w = int(block['Geometry']['BoundingBox']['Width'] * width)
            h = int(block['Geometry']['BoundingBox']['Height'] * height)
            return BoundingBox(x, y, w, h)
    
    return None

def parse_micr_string(micr_string: str) -> MICRData:
    # Remove all whitespace
    cleaned_string = re.sub(r"\s+", "", micr_string, flags=re.UNICODE)

    # Splitting MICR based on the special symbols 
    # However, Textract fails to identify special symbols sometimes. 
    # cleaned_string = cleaned_string.replace('⑆', ' ').replace('⑈', ' ').strip()
    # parts = cleaned_string.split(' ')

    # Splitting MICR based on any contguous sections of non-numeric characters. 
    cleaned_string = re.sub(r'\D+', ' ', cleaned_string).strip()
    parts = cleaned_string.split(' ')
    
    if len(parts) != 3:
        raise MICRExtractionError(
            f"`{micr_string}`: MICR string must contain exactly three parts separated by spaces after removing MICR symbols."
        )
    
    routing_number = parts[0]
    account_number = parts[1]
    check_number = parts[2]

    if not (routing_number.isnumeric() and account_number.isnumeric() and check_number.isnumeric()):
            raise MICRExtractionError(
                f"`{micr_string}`: MICR string must only contain numeric characters or MICR special characters."
            )
    
    return MICRData(routing_number, account_number, check_number)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_micr.py <image_path>")
        sys.exit(1)
    
    # Parse the image path from command-line arguments
    image_path = Path(sys.argv[1])

    session = boto3.Session(profile_name=AWS_PROFILE_NAME)
    textract_client = session.client('textract', region_name=AWS_REGION_NAME)
    
    # Call the function to extract bounding boxes
    micr_info = extract_micr_data(image_path, textract_client)
    print(micr_info)

    