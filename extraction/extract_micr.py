from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import sys

@dataclass
class MICRInfo:
    """ Represents the different MICR fields"""
    routing_number: int
    account_number: int
    check_number: int


def extract_micr(image_path: Path) -> MICRInfo:
    """ Extract MICR data from check image
    
    Args:
        image_path (Path): Path to check image
    
    Returns:
        MICRInfo: Class with the three fields contained in the MICR data
    
    """
    raise NotImplementedError("This function has not been implemented yet.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 extract_micr.py <image_path>")
        sys.exit(1)
    
    # Parse the image path from command-line arguments
    image_path = Path(sys.argv[1])
    
    # Call the function to extract bounding boxes
    micr_info = extract_micr(image_path)
    print(micr_info)
    
    