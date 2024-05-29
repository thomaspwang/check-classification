### DiversaTech x Sofi Spring 2024

# Check Classification and Fraud Detection

## Overview

## Repository Structure

## Environment Setup

You only need perform steps 1 through  4 **once**. You can check if you've completed the environment by checking that a `.venv/` file exists in your local repository. 


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
TODO: Install LLaVA @jerli

<br> **Step 7** <br>
For local testing and development, we recommend creating a local folder such as `sofi-check-classification/data` for PII images and labeled data.

## Usage Examples

<br> **Extracting Bounding Boxes** <br>
`python extract_bboxes.py ../data/mcd-test-3-front-images/mcd-test-3-front-93.jpg output_image.jpg`

<br> **Extracting MICR from an image** <br>
`python extract_micr.py ../data/mcd-test-3-front-images/mcd-test-3-front-93.jpg`

<br> **Treasury Check Classification** <br>
TODO: @Jerli

<br> **Extracting all data from a check image* <br>
`python extract.py ../data/mcd-test-3-front-images/mcd-test-3-front-93.jpg --model llava`


## Possible TO-DOs

- Configure environment variables automatically through `dotenv` or something instead of having redundant top-level variables such as `AWS_REGION` at the top of every file.
- Configure logging


## Debugging Tips

- Using `draw_bounding_boxes_on_image` in `extraction/extract_bboxes.py` can be useful for visualizing bounding boxes. Note that bounding box coordinates are specific to a particular image, so boxes can only be drawn on the images they were generated on.
- @jerli TODO