import cv2
import numpy as np
import boto3
import io
from PIL import Image

AWS_BUCKET_NAME = 'katie-sofi-bucket'
AWS_PROFILE_NAME = 'katiewang'
AWS_REGION_NAME = 'us-west-1'

@dataclass
class BoundingBox:
    # x and y represent the bottom left coordinate of the bounding box
    x: int
    y: int
    width: int
    height: int
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

def extract_bounding_boxes(profile, region, bucket_name, file_name) -> list[BoundingBox]:
    """ Extract bounding boxes from check image
    
    Args:
        profile: profile name on Amazon Web Services
        region: region the check image is stored at
        bucket_name: name of bucket storing check image
        file_name: file name of the check image

    Returns:
        list: list of coordinates for bounding box regions of interest

    Notes:
    The bounding box must include ALL the information pertaining to the region of interest,
    and have NO overlap with other regions of interest.
    
    """
    # Get the check which is stored in stevensegawa's bucket called aws-for-checks
    session = boto3.Session(profile_name=profile)
    s3_connection = session.resource('s3')
    client = session.client('textract', region_name=region)
    bucket = bucket_name
    document = file_name

    # Get the document from S3  
    s3_object = s3_connection.Object(bucket, document)
    s3_response = s3_object.get()
    stream = io.BytesIO(s3_response['Body'].read())
    image=Image.open(stream)

    # Detect text in the document
    # Process using S3 object
    response = client.detect_document_text(
        Document={'S3Object': {'Bucket': bucket, 'Name': document}})
    
    # Get the text blocks
    blocks=response['Blocks']
    boundingbox_list = []
    width, height = image.size 
    
    # Create image showing bounding box/polygon the detected lines/text
    for block in blocks:
            # Display information about a block returned by text detection
            if block['BlockType'] == 'LINE':
                print('Type: ' + block['BlockType'])
                if block['BlockType'] != 'PAGE':
                    print('Detected: ' + block['Text'])
                    print('Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")
                
                print('Id: {}'.format(block['Id']))
                if 'Relationships' in block:
                    print('Relationships: {}'.format(block['Relationships']))
                print('Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
                print('Polygon: {}'.format(block['Geometry']['Polygon']))
                x = block['Geometry']['BoundingBox']['Left'] * width
                y = block['Geometry']['BoundingBox']['Top'] * height
                w = block['Geometry']['BoundingBox']['Width'] * width
                h = block['Geometry']['BoundingBox']['Height'] * height
                boundingbox_list += [BoundingBox(x, y, w, h)]
                # draw=ImageDraw.Draw(image)
    return boundingbox_list


def get_image(profile, bucket_name, file_name) -> Image:
    """
    Retrieves the check image from AWS bucket.

    """
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
        rects: List of tuples (x, y, w, h) representing bounding boxes.
        max_center_distance: Maximum center distance to consider for merging.
        max_corner_distance: Maximum distance between corners to consider for merging.

    Returns:
        List of merged bounding boxes.
    """
    def corner_distance(box1, box2):
        """
        Calculate the distance between the corners of two bounding boxes.

        Args:
            box1: Tuple (x1, y1, w1, h1) representing the first box.
            box2: Tuple (x2, y2, w2, h2) representing the second box.

        Returns:
            Minimum distance between the corners.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the corners of each bounding box
        corners_box1 = [(x1, y1), (x1 + w1, y1), (x1, y1 + h1), (x1 + w1, y1 + h1)]
        corners_box2 = [(x2, y2), (x2 + w2, y2), (x2, y2 + h2), (x2 + w2, y2 + h2)]

        min_distance = float('inf')
        for corner1 in corners_box1:
            for corner2 in corners_box2:
                distance = np.sqrt((corner1[0] - corner2[0])**2 + (corner1[1] - corner2[1])**2)
                min_distance = min(min_distance, distance)

        return min_distance

    merged_rects = []
    for rect in rects:
        x, y, w, h = rect
        center_x = x + w // 2
        center_y = y + h // 2

        merged = False
        for i, merged_rect in enumerate(merged_rects):
            merged_x, merged_y, merged_w, merged_h = merged_rect
            merged_center_x = merged_x + merged_w // 2
            merged_center_y = merged_y + merged_h // 2

            center_distance = np.sqrt((center_x - merged_center_x)**2 + (center_y - merged_center_y)**2)
            corner_distance_val = corner_distance(rect, merged_rect)

            if center_distance < max_center_distance or corner_distance_val < max_corner_distance:
                # Merge by expanding the merged rectangle
                new_x = min(x, merged_x)
                new_y = min(y, merged_y)
                new_w = max(x + w, merged_x + merged_w) - new_x
                new_h = max(y + h, merged_y + merged_h) - new_y
                merged_rects[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break

        if not merged:
            merged_rects.append(rect)

    return merged_rects

def merge_overlapping_boxes(boxes):

    """
    Merge overlapping bounding boxes in a list.

    Args:
        boxes: List of tuples (x, y, w, h) representing bounding boxes.

    Returns:
        List of merged bounding boxes.
    """
    def calculate_iou(box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1: Tuple (x1, y1, w1, h1) representing the first box.
            box2: Tuple (x2, y2, w2, h2) representing the second box.

        Returns:
            IoU (Intersection over Union) value.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate the coordinates of the intersection rectangle
        x_intersection = max(x1, x2)
        y_intersection = max(y1, y2)
        w_intersection = min(x1 + w1, x2 + w2) - x_intersection
        h_intersection = min(y1 + h1, y2 + h2) - y_intersection

        if w_intersection <= 0 or h_intersection <= 0:

            return 0.0

        # Calculate the area of intersection rectangle
        intersection_area = w_intersection * h_intersection

        # Calculate the area of both bounding boxes
        area1 = w1 * h1
        area2 = w2 * h2

        # Calculate the Union area by using Formula: Union(A, B) = A + B - Inter(A, B)
        union_area = area1 + area2 - intersection_area

        # Calculate the Intersection over Union (IoU)
        iou = intersection_area / union_area
        return iou

    # Sort boxes based on their x-coordinate
    sorted_boxes = sorted(boxes, key=lambda x: x[0])

    merged_boxes = []
    while len(sorted_boxes) > 0:
        current_box = sorted_boxes.pop(0)
        x1, y1, w1, h1 = current_box
        merged_box = list(current_box)

        # Check if the current box overlaps with any other box
        overlapping_boxes = [box for box in sorted_boxes if calculate_iou(current_box, box) > 0]

        # Merge the overlapping boxes
        for box in overlapping_boxes:
            x2, y2, w2, h2 = box
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1 + w1, x2 + w2)
            y_max = max(y1 + h1, y2 + h2)
            merged_box = [x_min, y_min, x_max - x_min, y_max - y_min]
        
        merged_boxes.append(tuple(merged_box))
        # print(merged_boxes)
        # Remove the merged and overlapping boxes from the list
        sorted_boxes = [box for box in sorted_boxes if box not in overlapping_boxes]
        
    
    return merged_boxes

"""

Below are the parameters necessary to extract bounding boxes from a check.

profile: profile name on Amazon Web Services
region: region the check image is stored at
bucket_name: name of bucket storing check image
file_name: file name of the check image

"""

profile = 'stevensegawa'
region = 'us-west-1'
bucket_name = 'aws-for-checks'
file_name = 'warped_IMG_1599.jpg'

#Code to run the algorithm
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
    x, y, w, h = rect
    # print(x, y, w, h)
    cv2.rectangle(image, ((int)(x), (int)(y)), ((int)(x + w), (int)(y + h)), (0, 255, 0), 2)
for rect in micr_bounding_boxes:
    x, y, w, h = rect
    cv2.rectangle(image, ((int)(x), (int)(y)), ((int)(x + w), (int)(y + h)), (0, 255, 0), 2)

cv2.imshow("merged boxes", image)
cv2.imwrite("cooked.jpg", image)
cv2.waitKey(0) 
cv2.destroyAllWindows()