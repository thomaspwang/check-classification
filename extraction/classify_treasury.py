#!./venv/bin/python3.10
""" TODO: Module Docstring
"""

import argparse
import boto3
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from extract_bboxes import (
    extract_bounding_boxes_from_path,
)
from extract_handwriting import (
    parse_handwriting,
    Mode,
)
import time
from tqdm import tqdm


# AWS Session Config
AWS_PROFILE_NAME = 'thwang'
AWS_REGION_NAME = 'us-west-2'

EXTRACT_MODE = Mode.DOC_TR 

TREASURY_CHECK_NUMS = [70,  98, 132, 155, 163, 236, 287, 313, 331, 357, 362, 390, 406, 415,
       426, 439, 446, 494, 524, 644, 653, 668, 670, 671, 737, 777, 802, 837,
       847, 848, 858, 870, 951, 963, 967, 969]

def is_treasury_check(img_path: Path, textract_client) -> bool:
    #TODO: Docstring

    bounding_boxes = extract_bounding_boxes_from_path(img_path, textract_client)

    for bbox in bounding_boxes:
        data = parse_handwriting(img_path, bbox, EXTRACT_MODE).lower()


        print(data)
        if 'treasury' in data:
            return True

    return False


if __name__ == "__main__":
    session = boto3.Session(profile_name=AWS_PROFILE_NAME)
    textract_client = session.client('textract', region_name=AWS_REGION_NAME)
    
    file_path = Path(f"./data/images/mcd-test-4-front-{70}.jpg")
    is_treasury_check(file_path, textract_client)

    # num_total = len(TREASURY_CHECK_NUMS)
    # num_correct = 0

    # for treasury_check_num in tqdm(TREASURY_CHECK_NUMS, desc="Processing Checks"):
    #     file_path = Path(f"./data/images/mcd-test-4-front-{treasury_check_num}.jpg")

    #     if is_treasury_check(file_path, textract_client):
    #         num_correct += 1

    # print(f"Accuracy: {num_correct / num_total}%")
    # print(f"")

