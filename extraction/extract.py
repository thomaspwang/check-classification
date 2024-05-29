#!./venv/bin/python3.10
"""
Main script for extracting data from check images using AWS Textract and various OCR models.

This module orchestrates different extraction methods provided in the sofi-check-classification/extraction package 
to process check images, extract bounding boxes, merge nearby and overlapping boxes if specified, 
and then extract data from these boxes using specified a specified model.

Currently, you can use either LLaVA or docTR to extract data. However, as of the time of this commit,
doctr performs significantly worse than LLaVA.

Usage:
    python extract.py <img_path> --model <model>

Example:
    python extract.py ../data/mcd-test-3-front-images/mcd-test-3-front-93.jpg --model llava

Functions:
    extract_data(img_path: Path, textract_client, extract_mode: ExtractMode, merge_boxes: bool = False) -> str:
        Extracts data from a check image using specified extraction mode and optionally merges bounding boxes.


To run the script, ensure that AWS credentials are configured correctly.
"""

import argparse
import boto3
import cv2
from pathlib import Path
import numpy as np
from extract_bboxes import ( 
    extract_bounding_boxes_from_path,
    merge_nearby_boxes,
    merge_overlapping_boxes,
)
from extract_handwriting import (
    generate_doctr_model,
    generate_LLaVA_model,
    parse_handwriting,
    ExtractMode,
)
import time
from typing import Any

# AWS Session Config
AWS_PROFILE_NAME = 'thwang'
AWS_REGION_NAME = 'us-west-2'


def extract_data(
        img_path: Path,
        textract_client,
        extract_mode: ExtractMode,
        merge_boxes: bool = False,
) -> str:
    #TODO: Docstring
    start_time = time.time()

    bounding_boxes = extract_bounding_boxes_from_path(img_path, textract_client)

    checkpoint_1 = time.time()
    elapsed_1 = checkpoint_1 - start_time
    print(f"Bounding box extraction finished in {elapsed_1:.2f} seconds")

    if merge_boxes:
        image = cv2.imread(str(img_path))
        max_distance = 20
        max_corner = int(image.shape[0] * 0.02)

        merged_bboxes = merge_nearby_boxes(bounding_boxes[:-2], max_distance, max_corner)
        overlapped_merged = merge_overlapping_boxes(merged_bboxes)

        while overlapped_merged != merge_overlapping_boxes(overlapped_merged):
            merged_bboxes = merged_bboxes + overlapped_merged
            overlapped_merged = merge_overlapping_boxes(overlapped_merged)

        bounding_boxes = overlapped_merged

    checkpoint_2 = time.time()
    elapsed_2 = checkpoint_2 - checkpoint_1
    print(f"Bounding box merging finished in {elapsed_2:.2f} seconds")

    model: Any
    if extract_mode == ExtractMode.DOC_TR:
        model = generate_doctr_model()
    elif extract_mode == ExtractMode.LLAVA:
        model = generate_LLaVA_model()

    data = [parse_handwriting(img_path, bbox, extract_mode, model) for bbox in bounding_boxes]

    checkpoint_3 = time.time()
    elapsed_3 = checkpoint_3 - checkpoint_2
    print(f"Bounding box parsing finished in {elapsed_3:.2f} seconds")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='File path containing the image.')
    parser.add_argument('-m', '--model', choices=['llava', 'doctr'], required=True, help="Specify 'llava' or 'doctr' as the model to use.")
    args = parser.parse_args()
    
    img_path: Path = Path(args.img_path)

    selected_mode: ExtractMode
    match args.model:
        case 'llava':
            selected_mode = ExtractMode.LLAVA
        case 'doctr':
            selected_mode = ExtractMode.DOC_TR

    print(f"Extracting data from {img_path}")

    session = boto3.Session(profile_name=AWS_PROFILE_NAME)
    textract_client = session.client('textract', region_name=AWS_REGION_NAME)
    
    data = extract_data(img_path, textract_client, selected_mode)

    print("Done. Extracted data:\n")
    print(data)
    
