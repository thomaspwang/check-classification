### DiversaTech x Sofi Spring 2024

# Check Classification and Fraud Detection

## Overview

This repo provides a set of modules and functions to extract information from check images as well as classify check types.
It supports different OCR strategies including LLaVA, DocTR and Amazon Textract.

https://docs.google.com/presentation/d/1lbg0_yc7BgP9efKZR0iPUIo8Nylqx6sAo6SApx0MH-E/edit?pli=1#slide=id.g2e0ac83555e_0_0

## Environment Setup

<br> **Step 0** <br>
We used AWS SageMaker notebooks throughout the development of this project. Ensure that your notebook is using at least a `ml.g5.2xlarge` instance. However, to run models in their non-quantized form, you must use at least a `ml.g5.12xlarge` instance. Lastly, we recommend using a volume size of atleast 500GB EBS since the LLaVA model is large.

<br> **Step 1** <br>
Make sure you have an updated Python version with
```
python --version  # should be 3.10 or higher
```

<br> **Step 2** <br>
Navigate to the root of this repository (`sofi-check-classification`) and create a new virtual environment with either
```
python -m venv venv
# or
python3 -m venv venv
```

<br> **Step 3** <br>
Activate the venv with
```
source venv/bin/activate
```

<br> **Step 4** <br>
Install requirements with
```
pip install -r requirements.txt
```

<br> **Step 5** <br>
Even after installing `requirements.txt`, you may need need to manually install these:

```
pip install python-doctr
pip install torch torchvision torchaudio
```

Note: We've found that we needed to do this on EC2 instances (`ml.g5.12xlarge`).
Note: `pip install torch torchvision torchaudio` assumes that your system has Cuda 12.1.

<br> **Step 6** <br>
To install llava, run
```
pip install -e git+https://github.com/haotian-liu/LLaVA.git@c121f04#egg=llava
```

<br> **Step 7** <br>
Once within the notebook terminal, run 
```
export HOME="/home/ec2-user/SageMaker"
```
This is because the 500GB drive is mounted on SageMaker, while a smaller 50GB drive is mounted at /home/ec2-user.
An out-of-disk-space error will occur as libraries and model weights will download to the default HOME=/home/ec2-user.

<br> **Step 8** <br>
For local testing and development, we recommend creating a local folder such as `sofi-check-classification/data` for PII images and labeled data.

## Usage Example
```python
import boto3
from classify_treasury import is_treasury_check
from extract_micr import extract_micr_data, MICRData, MICRExtractionError
from extract import extract_data
from extract_bboxes import BoundingBox
from parse_bbox import parse_bbox, ExtractMode, generate_LLaVA_model

AWS_PROFILE_NAME = ...
AWS_AWS_REGION_NAME = ...

INPUT_IMAGE_PATH = Path("...")

# Generating Models
llava_model = generate_LLaVA_model()

session = boto3.Session(profile_name=AWS_PROFILE_NAME)
textract_client = session.client('textract', region_name=AWS_AWS_REGION_NAME)

# Classifying Treasury Checks
is_treasury_check: bool = is_treasury_check(INPUT_IMAGE_PATH, llava_model)

# Extracting MICR data
try:
    micr_data: MICRData = extract_micr_data(INPUT_IMAGE_PATH, textract_client)
except MICRExtractionError as e:
    raise

# Scraping specific check data using LLaVA
PROMPT = "Scan this check and output the check amount as a string"
llava_check_amount_output: str = parse_bbox(INPUT_IMAGE_PATH, box=None, ExtractMode.LLAVA, llava_model, PROMPT)

# Scraping all check data using LLaVA and doctr
check_data_doctr: list[str] = extract_data(INPUT_IMAGE_PATH, textract_client, ExractMode.DOC_TR)
check_data_llava: list[str] = extract_data(INPUT_IMAGE_PATH, textract_client, ExractMode.LLAVA)
```

## Demos

<br> **Extracting Bounding Boxes** <br>
Writes a full-sized check image with the bounding boxes draw on it to a specified output file.<br>
`python extract_bboxes.py ../data/mcd-test-3-front-images/mcd-test-3-front-93.jpg output_image.jpg`

<br> **Extracting MICR from an image** <br>
Prints out a `MICRData` dataclass object generated from a full-sized check image to the console.<br>
`python extract_micr.py ../data/mcd-test-3-front-images/mcd-test-3-front-93.jpg`

<br> **Treasury Check Classification** <br>
Prints out whether or not a given full-sized input check is a treasury check or not.<br>
`python classify_treasury.py ../data/mcd-test-4-front-images/mcd-test-4-front-70.jpg`

<br> **Extracting all data from a check image** <br>
Prints out all text data extracted from a full-sized check image as a list of strings.<br>
`python extract.py ../data/mcd-test-3-front-images/mcd-test-3-front-93.jpg --model llava`

## Benchmarking Algorithms

<br> **Add or a Select a Strategy** <br>
Go to extraction/analyze_checks.py, and follow the module docstring to add a strategy or select
an existing strategy of {LLAVA_AMOUNT_AND_NAME, TEXTRACT_MICR, LLAVA_TREASURY}. 

<br> **Obtain a Dataset and Labels** <br>
Obtain a dataset of check images and put it in a folder. The image file names must be in the format
`mcd-test-N-front-##.jpg`, where N is one digit and ## can be any amount of digits. 

The label file should be a csv with each row corresponding directly to the check number (e.g. row 0 are the headers,
and row 1 corresponds to `mcd-test-N-front-1.jpg`. The label file must be contiguous, but the dataset
does not need to include all files specified by the label set. The datasets provided by @jts (N=1-5)
should all work. Examples of label files are in the datasets provided by @jts; convert the mcd-test-N-image-details.numbers
to csvs and that will be a valid label file.

<br> **Run Inference on the Dataset** <br>
`cd sofi-check-classification`
`python extraction/analyze_checks.py <path_to_dataset_folder> <path_to_output_csv> <strategy>`

example:
`python extraction/analyze_checks.py ../mcd-test-3-front-images/ ../LLAVA_TREASURY_PREDICTIONS.csv LLAVA_TREASURY`
This will compute some statistics like seconds / inference and $ / inference.

<br> **Compute Aggregate Statistics** <br>
To compute statistics like hit rate, accuracy and edit distance, run
`python scripts/compare_predictions_to_labels.py <path_to_dataset_folder> <path_to_prediction_csv> <path_to_labels> --verbose`

example:
`python scripts/compare_predictions_to_labels.py ../mcd-test-3-front-images/ ../LLAVA_TREASURY_PREDICTIONS.csv ../mcd-test-3-image-details.csv --verbose`

## Possible TO-DOs

- Configure environment variables automatically through `dotenv` or something instead of having redundant top-level variables such as `AWS_REGION` at the top of every file.
- Configure logging


## Debugging Tips

- Using `draw_bounding_boxes_on_image` in `extraction/extract_bboxes.py` can be useful for visualizing bounding boxes. Note that bounding box coordinates are specific to a particular image, so boxes can only be drawn on the images they were generated on.