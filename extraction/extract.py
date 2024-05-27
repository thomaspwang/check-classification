#!./venv/bin/python3.10
""" Main script for extracting data from check images. 

This module is responsible for orchestrating the different extraction methods 
in the extraction package.
"""

import argparse
import cv2
from pathlib import Path
import numpy as np
from extract_bboxes import (
    BoundingBox, 
    extract_bounding_boxes_from_path,
    merge_nearby_boxes,
    merge_overlapping_boxes,
)
from extract_handwriting import (
    parse_handwriting,
    Mode,
)

EXTRACT_MODE = Mode.DOC_TR  # TODO: Probably add as a command line input later

def extract_data(
        img_path: Path,
        merge_boxes: bool = False
) -> str:
    #TODO: Docstring

    image = cv2.imread(str(img_path))

    max_distance = 20
    max_corner = int(image.shape[0] * 0.02)

    bounding_boxes = extract_bounding_boxes_from_path(img_path)

    if merge_boxes:
        merged_bboxes = merge_nearby_boxes(bounding_boxes[:-2], max_distance, max_corner)
        overlapped_merged = merge_overlapping_boxes(merged_bboxes)

        while overlapped_merged != merge_overlapping_boxes(overlapped_merged):
            merged_bboxes = merged_bboxes + overlapped_merged
            overlapped_merged = merge_overlapping_boxes(overlapped_merged)

        bounding_boxes = overlapped_merged

    data = [parse_handwriting(img_path, bbox, EXTRACT_MODE) for bbox in bounding_boxes]

    return data


if __name__ == "__main__":
    # TODO: Improve cmd interface

    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='TODO')
    args = parser.parse_args()\
    
    img_path: Path = Path(args.img_path)

    print(f"Extracting data from {img_path}")
    data = extract_data(img_path)
    print("Done. Extracted data:\n")
    print(data)
    
