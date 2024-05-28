import boto3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from PIL import Image
import re
import tempfile
import sys
from extract_bboxes import extract_bounding_boxes_from_path, BoundingBox
from extraction_utils import crop_image, merge_two_bounding_boxes

AWS_PROFILE_NAME = 'thwang'
AWS_REGION_NAME = 'us-west-2'

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
        client: An AWS boto3 textract_client.
    
    Returns:
        MICRInfo: Class with the three fields contained in the MICR data
        None: Returns None if cannot parse MICR data.
    
    """
    bounding_boxes: list[BoundingBox] = extract_bounding_boxes_from_path(image_path, textract_client)

    # MICR is bounded two boxes typically.
    num_bounding_boxes = len(bounding_boxes)
    if num_bounding_boxes < 2:
        raise MICRExtractionError(
            f"There should be more than two bounding boxes that are extracted from the check at {image_path}.\
            Instead, there are {num_bounding_boxes}."
        )

    micr_bounding_box = merge_two_bounding_boxes(bounding_boxes[-2:])

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file_path = Path(temp_file.name)
        image: Image = crop_image(image_path, micr_bounding_box)

        # TODO: It would probably be more efficient to create a BytesIO object sending image data to Textract
        #       instead of using a tempfile. But it works for now.
        image.save(temp_file_path)

        with temp_file_path.open(mode="rb") as f:
            response = textract_client.detect_document_text(
                Document={'Bytes': f.read()}
            )

    blocks = response['Blocks']
    micr_string: str = ""
        
    for block in blocks:
        if block['BlockType'] == 'LINE':
            if micr_string != "":
                raise MICRExtractionError(
                    f"There should be only be one textstring being extracted from the MICR box.\
                    In addition to '{micr_string}', '{block['Text']}' was also extracted."
                )
            
            micr_string = block['Text']


    micr_data: MICRData = parse_micr_string(micr_string)
    return micr_data

def parse_micr_string(micr_string: str) -> MICRData:
    # Remove all whitespace
    cleaned_string = re.sub(r"\s+", "", micr_string, flags=re.UNICODE)

    # Splitting MICR based on the special symbols
    cleaned_string = cleaned_string.replace('⑆', ' ').replace('⑈', ' ').strip()
    parts = cleaned_string.split(' ')
    
    if len(parts) != 3:
        print(parts)
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
    print(hash(micr_info))
    