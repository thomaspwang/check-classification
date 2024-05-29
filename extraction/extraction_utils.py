"""
Utility functions for extracting and manipulating bounding boxes in images.

This module provides helper functions to crop images based on bounding boxes,
merge multiple bounding boxes, find the largest bounding box by area, and
stretch a bounding box by a given percentage.

Functions:
    crop_image(image_path: Path, bbox: BoundingBox) -> Image:
        Crops an image to the specified bounding box.

    merge_n_bounding_boxes(boxes: list[BoundingBox]) -> BoundingBox:
        Merges a list of bounding boxes into a single bounding box.

    merge_two_bounding_boxes(boxes: list[BoundingBox]) -> BoundingBox:
        Merges exactly two bounding boxes into one.

    largest_area_bounding_box(boxes: list[BoundingBox]) -> BoundingBox:
        Returns the bounding box with the largest area.

    stretch_bounding_box(bbox: BoundingBox, percentage: float = 0.03) -> BoundingBox:
        Stretches a bounding box by a given percentage in all directions.
"""

from pathlib import Path
from extract_bboxes import BoundingBox
from PIL import Image

def crop_image(image_path: Path, bbox: BoundingBox) -> Image:
    """ Crops the image at image_path bounding by bbox and returns a Pillow Image

    If bbox is None, then no cropping occurs
    """
    image = Image.open(image_path)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    if bbox is None:
        return image

    x1, y1 = bbox.top_left_corner()
    x2, y2 = bbox.bottom_right_corner()
    return image.crop((x1, y1, x2, y2))

def merge_n_bounding_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    assert len(boxes) > 0, "The list of bounding boxes should not be empty."

    x_min = boxes[0].x
    y_min = boxes[0].y
    x_max = boxes[0].x + boxes[0].width
    y_max = boxes[0].y + boxes[0].height
    
    # Loop through the remaining boxes to find the outermost coordinates
    for box in boxes[1:]:
        x_min = min(x_min, box.x)
        y_min = min(y_min, box.y)
        x_max = max(x_max, box.x + box.width)
        y_max = max(y_max, box.y + box.height)
    
    # Calculate the new width and height
    new_width = x_max - x_min
    new_height = y_max - y_min
    
    return BoundingBox(x=x_min, y=y_min, width=new_width, height=new_height)

def merge_two_bounding_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    assert len(boxes) == 2

    box1, box2 = boxes
    
    # Calculate the coordinates of the top-left corner of the merged bounding box
    x_min = min(box1.x, box2.x)
    y_min = min(box1.y, box2.y)
    
    # Calculate the coordinates of the bottom-right corner of the merged bounding box
    x_max = max(box1.x + box1.width, box2.x + box2.width)
    y_max = max(box1.y + box1.height, box2.y + box2.height)
    
    new_width = x_max - x_min
    new_height = y_max - y_min
    
    return BoundingBox(x=x_min, y=y_min, width=new_width, height=new_height)

def largest_area_bounding_box(boxes: list[BoundingBox]) -> BoundingBox:
    """ Returns the bounding box with the largest area from a list of bounding boxes.
    """
    if not boxes:
        raise ValueError("The list of bounding boxes is empty.")

    largest_box = max(boxes, key=lambda box: box.width * box.height)
    return largest_box


def stretch_bounding_box(bbox: BoundingBox, percentage: float = 0.03) -> BoundingBox:
    """ Stretches a bounding box by a given percentage in all directions.
    """
    # Calculate the amount to stretch
    stretch_x = int(bbox.width * percentage)
    stretch_y = int(bbox.height * percentage)

    # Calculate the new dimensions
    new_x = bbox.x - stretch_x
    new_y = bbox.y - stretch_y
    new_width = bbox.width + (2 * stretch_x)
    new_height = bbox.height + (2 * stretch_y)

    return BoundingBox(x=new_x, y=new_y, width=new_width, height=new_height)