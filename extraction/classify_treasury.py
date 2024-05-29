#!./venv/bin/python3.10
""" This script checks a check is an United States Treasury check.

Usage:
    python classify_treasury.py <path_to_check_image>

Example:
    python classify_treasury.py ../data/mcd-test-3-front-images/mcd-test-3-front-93.jpg

Functions:
    is_treasury_check(img_path: Path, model: any) -> bool:
        Returns true if the image is a treasury check, false if it is not.
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
    generate_LLaVA_model,
    parse_handwriting,
    ExtractMode,
)
from extraction_utils import crop_image
from PIL import Image
import time
from tqdm import tqdm

TREASURY_CHECK_NUMS = [70,  98, 132, 155, 163, 236, 287, 313, 331, 357, 362, 390, 406, 415,
       426, 439, 446, 494, 524, 644, 653, 668, 670, 671, 737, 777, 802, 837,
       847, 848, 858, 870, 951, 963, 967, 969]

def is_treasury_check(img_path: Path, model: any) -> bool:
    # Uses LLAVA to see if UNITED STATES TREASURY is present in the check.

    PROMPT = "Is the word \"UNITED STATES TREASURY\" written in a Gothic / Old English font present on this check? Only answer one word: True or False."
    output = parse_handwriting(img_path, None, ExtractMode.LLAVA, model, PROMPT)
    if output.upper() == "TRUE":
        return True
    elif output.upper() == "FALSE":
        return False
    else:
        raise ValueError(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Folder path of dataset 4.')
    args = parser.parse_args()

    model = generate_LLaVA_model()

    print(f"This file is a treasury check: {is_treasury_check(args.image_path, model)}")