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
    # TODO: Probably clean this!
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='Folder path of dataset 4.')
    args = parser.parse_args()

    num_total = len(TREASURY_CHECK_NUMS)
    num_correct = 0

    model = generate_LLaVA_model()
    for treasury_check_num in tqdm(TREASURY_CHECK_NUMS, desc="Processing Treasury Checks"):
        file_path = Path(f"{args.dataset_folder}/mcd-test-4-front-{treasury_check_num}.jpg")

        if is_treasury_check(file_path, model):
            num_correct += 1

    print(f"Accuracy: {num_correct / num_total * 100}%")
    print(f"")
