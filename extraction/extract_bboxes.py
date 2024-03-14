from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import sys

class RegionType(Enum):
    """ Enum for check region type 
    
    Example Usage:
        >>> bounding_box = BoundingBox(0, 0, 100, 100, RegionType.AMOUNT)
    """
    DATE = 0
    AMOUNT = 1
    PAYEE = 2
    MEMO = 3

@dataclass
class BoundingBox:
    """ Represent bounding boxes around regions of interest"""
    # Top left corner coordinates
    min_x: int
    min_y: int

    # Top right corner coodinates
    max_x: int
    min_y: int

    region_type: RegionType


def extract_bounding_boxes(image_path: Path) -> dict[RegionType, BoundingBox]:
    """ Extract bounding boxes from check image
    
    Args:
        image_path (Path): Path to check image
    
    Returns:
        dict[RegionType, BoundingBox]: Dict mapping regions of interest to bounding boxes

    Notes:
    The bounding box must include ALL the information pertaining to the region of interest,
    and have NO overlap with other regions of interest.
    
    """
    raise NotImplementedError("This function has not been implemented yet.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 bbox_extract.py <image_path>")
        sys.exit(1)
    
    # Parse the image path from command-line arguments
    image_path = Path(sys.argv[1])
    
    # Call the function to extract bounding boxes
    bounding_boxes, region_to_bbox = extract_bounding_boxes(image_path)
    
    # Load the image using OpenCV
    image = cv2.imread(str(image_path))
    
    # Draw bounding boxes and label them
    for region, bbox in bounding_boxes.items():
        cv2.rectangle(image, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), (0, 255, 0), 2)
        cv2.putText(image, region.value, (bbox.x_min, bbox.y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the image
    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()