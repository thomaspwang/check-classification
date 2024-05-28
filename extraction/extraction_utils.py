from pathlib import Path
from extract_bboxes import BoundingBox
from PIL import Image

def crop_image(image_path: Path, bbox: BoundingBox):
    image = Image.open(image_path)

    if bbox is None:
        return image
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    x1, y1 = bbox.top_left_corner()
    x2, y2 = bbox.bottom_right_corner()
    return image.crop((x1, y1, x2, y2))

def merge_two_bounding_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    assert len(boxes) == 2

    box1, box2 = boxes
    
    # Calculate the coordinates of the top-left corner of the merged bounding box
    x_min = min(box1.x, box2.x)
    y_min = min(box1.y, box2.y)
    
    # Calculate the coordinates of the bottom-right corner of the merged bounding box
    x_max = max(box1.x + box1.width, box2.x + box2.width)
    y_max = max(box1.y + box1.height, box2.y + box2.height)
    
    # Calculate the new width and height
    new_width = x_max - x_min
    new_height = y_max - y_min
    
    # Create and return the merged bounding box
    return BoundingBox(x=x_min, y=y_min, width=new_width, height=new_height)
