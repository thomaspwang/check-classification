from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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

