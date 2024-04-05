from enum import Enum
from pathlib import Path
from extract_bboxes import BoundingBox, RegionType

class Mode(Enum):
    DOC_TR = "DocTR"
    TR_OCR = "TrOCR"
    LLAVA = "LLAVA"
    AMAZON_TEXTRACT = "Amazon Textract"
    # add more as needed

def parse_handwriting(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
        mode : Mode,
) -> str:
    match mode:
        case Mode.DOC_TR:
            return parse_handwriting_doctr(img_path, region, box)
        case Mode.TR_OCR:
            return parse_handwriting_trocr(img_path, region, box)
        case Mode.LLAVA:
            return parse_handwriting_llava(img_path, region, box)
        case Mode.AMAZON_TEXTRACT:
            return parse_handwriting_amazon_textract(img_path, region, box)
        case _:
            raise ValueError(f"Invalid mode: {mode}")


def parse_handwriting_doctr(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using DocTR model """
    raise NotImplementedError("This function has not been implemented yet.")


def parse_handwriting_trocr(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using TrOCR model """
    raise NotImplementedError("This function has not been implemented yet.")


# def parse_handwriting_llava(
#         img_path: Path,
#         region : RegionType,
#         box : BoundingBox,
# ) -> str:
#     """ Parse handwriting using LLAVA model """
#     from gradio_client import Client
#     # load Client
#     client = Client("https://llava.hliu.cc/")
#     # submit prompt including image and text
#     result = client.predict(
#         "Please tell me the name of the payee, the amount paid on the check, the date marked, and the address/memo on this check.",	# str in Textbox component
#         img_path, # filepath for Image component
#         "Default", # Literal['Crop', 'Resize', 'Pad', 'Default']  in 'Preprocess for non-square image' Radio component
#         api_name="/add_text" # choose endpoint
    # # )
    # result = client.predict(
    #     "llava-v1.6-34b",
    #     .1, # float (numeric value between 0.0 and 1.0) in 'Temperature' Slider component
    #     .3, # float (numeric value between 0.0 and 1.0) in 'Top P' Slider component
    #     150, # float (numeric value between 0 and 1024) in 'Max output tokens' Slider component
    #     api_name="/http_bot" # choose endpoint
    # )
    # print('\n\n\nRESULTS:')
    # print(result[0][1])
    
import boto3
import io
from PIL import Image, ImageDraw

def parse_handwriting_amazon_textract(
        img_path: Path,
        region : RegionType,
        box : BoundingBox,
) -> str:
    """ Parse handwriting using Amazon Textract """

    # Get the check which is stored in stevensegawa's bucket called aws-for-checks
    session = boto3.Session(profile_name='katiewang')
    s3_connection = session.resource('s3')
    client = session.client('textract', region_name='us-west-1')
    bucket = 'katie-sofi-bucket'
    document = 'warped_IMG_1599.jpg'
    
    # Get the document from S3  
    s3_object = s3_connection.Object(bucket, document)
    s3_response = s3_object.get()
    stream = io.BytesIO(s3_response['Body'].read())
    image=Image.open(stream)

    #To process using image bytes:                      
    #image_binary = stream.getvalue()
    #response = client.detect_document_text(Document={'Bytes': image_binary})

    # Detect text in the document
    # Process using S3 object
    response = client.detect_document_text(
        Document={'S3Object': {'Bucket': bucket, 'Name': document}})

    # Get the text blocks
    blocks=response['Blocks']

    width, height = image.size 
    print("IMPORTANT:", width, height)   
    print ('Detected Document Text')
    boundingbox_list = []
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
                boundingbox_list += [(x, y, w, h)]
                draw=ImageDraw.Draw(image)

            # Draw WORD - Green -  start of word, red - end of word
            """
            if block['BlockType'] == "WORD":
                draw.line([(width * block['Geometry']['Polygon'][0]['X'],
                height * block['Geometry']['Polygon'][0]['Y']),
                (width * block['Geometry']['Polygon'][3]['X'],
                height * block['Geometry']['Polygon'][3]['Y'])],fill='green',
                width=2)
            
                draw.line([(width * block['Geometry']['Polygon'][1]['X'],
                height * block['Geometry']['Polygon'][1]['Y']),
                (width * block['Geometry']['Polygon'][2]['X'],
                height * block['Geometry']['Polygon'][2]['Y'])],
                fill='red',
                width=2)  
            """
  
            # Draw box around entire LINE
            if block['BlockType'] == "LINE":
                points=[]

                for polygon in block['Geometry']['Polygon']:
                    points.append((width * polygon['X'], height * polygon['Y']))
                draw.polygon((points), outline='red')
    print(boundingbox_list)    

    # Display the image
    image.show()
    block_count = len(blocks)
    print("Blocks detected: " + str(block_count))

"""
# Practice Running LLAVA Model
dummy = Mode('LLAVA')
parse_handwriting(str(Path('boundedboxcheck.png')), None, None, dummy)
"""

# Practice running Amazon Textract Model
dummy2 = Mode("Amazon Textract")
parse_handwriting(None, None, None, dummy2)