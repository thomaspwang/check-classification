import cv2
import numpy as np
import boto3
import io
from PIL import Image
from dataclasses import dataclass

@dataclass
class BoundingBox:
    """
    Represents a rectangular bounding box.

    Attributes:
        x (int): The x-coordinate of the top-left corner of the bounding box.
        y (int): The y-coordinate of the top-left corner of the bounding box.
        width (int): The width of the bounding box.
        height (int): The height of the bounding box.
    """
    x: int
    y: int
    width: int
    height: int

    def top_left_corner(self):
        """Return the top-left corner of the bounding box."""
        return (self.x, self.y)

    def top_right_corner(self):
        """Return the top-right corner of the bounding box."""
        return (self.x + self.width, self.y)

    def bottom_left_corner(self):
        """Return the bottom-left corner of the bounding box."""
        return (self.x, self.y + self.height)

    def bottom_right_corner(self):
        """Return the bottom-right corner of the bounding box."""
        return (self.x + self.width, self.y + self.height)

def extract_bounding_boxes(profile, region, bucket_name, file_name) -> list[BoundingBox]:
    """ Extract bounding boxes from check image
    
    Args:
        profile: profile name on Amazon Web Services
        region: region the check image is stored at
        bucket_name: name of bucket storing check image
        file_name: file name of the check image

    Returns:
        list[BoundingBox]: list of bounding boxes with coordinates and dimensions
    """
    session = boto3.Session(profile_name=profile)
    s3_connection = session.resource('s3')
    client = session.client('textract', region_name=region)
    bucket = bucket_name
    document = file_name

    s3_object = s3_connection.Object(bucket, document)
    s3_response = s3_object.get()
    stream = io.BytesIO(s3_response['Body'].read())
    image = Image.open(stream)

    response = client.detect_document_text(
        Document={'S3Object': {'Bucket': bucket, 'Name': document}})
    
    blocks = response['Blocks']
    boundingbox_list = []
    width, height = image.size 
    print("IMPORTANT:", width, height)   
    print('Detected Document Text')
    
    for block in blocks:
        if block['BlockType'] == 'LINE':
            x = int(block['Geometry']['BoundingBox']['Left'] * width)
            y = int(block['Geometry']['BoundingBox']['Top'] * height)
            w = int(block['Geometry']['BoundingBox']['Width'] * width)
            h = int(block['Geometry']['BoundingBox']['Height'] * height)
            bbox = BoundingBox(x, y, w, h)
            boundingbox_list.append(bbox)
    return boundingbox_list

def get_image(profile, bucket_name, file_name) -> Image:
    # Get the check which is stored in bucket_name
    session = boto3.Session(profile_name=profile)
    s3_connection = session.resource('s3')
    bucket = bucket_name
    document = file_name

    # Get the document from S3  
    s3_object = s3_connection.Object(bucket, document)
    s3_response = s3_object.get()
    stream = io.BytesIO(s3_response['Body'].read())
    image=Image.open(stream)

    return image
    
def merge_nearby_boxes(rects, max_center_distance, max_corner_distance):
    """
    Merge nearby bounding boxes in a list based on both center distance and distance between corners.

    Args:
        rects: List of BoundingBox objects representing bounding boxes.
        max_center_distance: Maximum center distance to consider for merging.
        max_corner_distance: Maximum distance between corners to consider for merging.

    Returns:
        List of merged BoundingBox objects.
    """
    def corner_distance(box1, box2):
        """
        Calculate the distance between the corners of two bounding boxes.

        Args:
            box1: BoundingBox representing the first box.
            box2: BoundingBox representing the second box.

        Returns:
            Minimum distance between the corners.
        """
        corners_box1 = [box1.top_left_corner(), box1.top_right_corner(), box1.bottom_left_corner(), box1.bottom_right_corner()]
        corners_box2 = [box2.top_left_corner(), box2.top_right_corner(), box2.bottom_left_corner(), box2.bottom_right_corner()]

        min_distance = float('inf')
        for corner1 in corners_box1:
            for corner2 in corners_box2:
                distance = np.sqrt((corner1[0] - corner2[0])**2 + (corner1[1] - corner2[1])**2)
                min_distance = min(min_distance, distance)

        return min_distance

    merged_rects = []
    for rect in rects:
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        center_x = x + w // 2
        center_y = y + h // 2

        merged = False
        for i, merged_rect in enumerate(merged_rects):
            merged_x, merged_y, merged_w, merged_h = merged_rect.x, merged_rect.y, merged_rect.width, merged_rect.height
            merged_center_x = merged_x + merged_w // 2
            merged_center_y = merged_y + merged_h // 2

            center_distance = np.sqrt((center_x - merged_center_x)**2 + (center_y - merged_center_y)**2)
            corner_distance_val = corner_distance(rect, merged_rect)

            if center_distance < max_center_distance or corner_distance_val < max_corner_distance:
                new_x = min(x, merged_x)
                new_y = min(y, merged_y)
                new_w = max(x + w, merged_x + merged_w) - new_x
                new_h = max(y + h, merged_y + merged_h) - new_y
                merged_rects[i] = BoundingBox(new_x, new_y, new_w, new_h)
                merged = True
                break

        if not merged:
            merged_rects.append(rect)

    return merged_rects

def merge_overlapping_boxes(boxes):
    """
    Merge overlapping bounding boxes in a list.

    Args:
        boxes: List of BoundingBox objects representing bounding boxes.

    Returns:
        List of merged BoundingBox objects.
    """
    def calculate_iou(box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1: BoundingBox representing the first box.
            box2: BoundingBox representing the second box.

        Returns:
            IoU (Intersection over Union) value.
        """
        x1, y1, w1, h1 = box1.x, box1.y, box1.width, box1.height
        x2, y2, w2, h2 = box2.x, box2.y, box2.width, box2.height

        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        w_intersection = min(x1 + w1, x2 + w2) - x_intersection
        h_intersection = min(y1 + h1, y2 + h2) - y_intersection

        if w_intersection <= 0 or h_intersection <= 0:
            return 0.0

        intersection_area = w_intersection * h_intersection
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersection_area

        return intersection_area / union_area

    sorted_boxes = sorted(boxes, key=lambda box: box.x)

    merged_boxes = []
    while sorted_boxes:
        current_box = sorted_boxes.pop(0)
        x1, y1, w1, h1 = current_box.x, current_box.y, current_box.width, current_box.height
        merged_box = BoundingBox(x1, y1, w1, h1)

        overlapping_boxes = [box for box in sorted_boxes if calculate_iou(current_box, box) > 0]

        for box in overlapping_boxes:
            x2, y2, w2, h2 = box.x, box.y, box.width, box.height
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1 + w1, x2 + w2)
            y_max = max(y1 + h1, y2 + h2)
            merged_box = BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)

        merged_boxes.append(merged_box)
        sorted_boxes = [box for box in sorted_boxes if box not in overlapping_boxes]

    return merged_boxes

"""

Below are the parameters necessary to extract bounding boxes from a check.

profile: profile name on Amazon Web Services
region: region the check image is stored at
bucket_name: name of bucket storing check image
file_name: file name of the check image

"""

# profile = 'stevensegawa'
# region = 'us-west-1'
# bucket_name = 'aws-for-checks'
# file_name = 'warped_IMG_1599.jpg'

profile = 'christinayue'
region = 'us-west-1'
bucket_name = 'aws-bboxes-checks'
file_name = 'inputcheck.jpg'

# Code to run the algorithm
bucket_image = get_image(profile, bucket_name, file_name)
image = cv2.cvtColor(np.array(bucket_image), cv2.COLOR_RGB2BGR)
max_distance = 20
max_corner = (int)(image.shape[0] * 0.02)
bounding_boxes = extract_bounding_boxes(profile, region, bucket_name, file_name)
micr_bounding_boxes = bounding_boxes[-2:]
merged_rects = merge_nearby_boxes(bounding_boxes[:-2], max_distance, max_corner)
overlapped_merged = merge_overlapping_boxes(merged_rects)

while overlapped_merged != merge_overlapping_boxes(overlapped_merged):
    merged_rects = merged_rects + overlapped_merged
    overlapped_merged = merge_overlapping_boxes(overlapped_merged)

# Draw merged rectangles on the image
for rect in overlapped_merged:
    # Updated to use the attributes directly instead of unpacking
    x, y, w, h = rect.x, rect.y, rect.width, rect.height
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

for rect in micr_bounding_boxes:
    # Updated to use the attributes directly instead of unpacking
    x, y, w, h = rect.x, rect.y, rect.width, rect.height
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

scale_percent = 50  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Resized Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()