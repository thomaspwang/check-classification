#!./venv/bin/python3.10
""" Script for analyzing a folder of checks specifying an extraction strategy to a csv.

Usage:
    ./analyze_checks.py <path_to_data_folder> <path_to_output_csv> strategy>

Example:
    chmod +x analyze_checks.py
    ./analyze_checks.py ./data/mcd-test-3-front-images results.csv TEXTRACT_MICR

Arguments:
    data_folder_path: Input folder of check images.
    out_file: Output .csv file.
    strategy: Specify a strategy. Choices are: LLAVA_AMOUNT_AND_NAME, TEXTRACT_MICR
    hourly_cost: This flag defaults to 5.672, the price of a ml.g5.12xlarge on demand pricing.
                 Adjust it to your instance cost to accurately calculate cost / instance.

    
How to Add A New Strategy:
    1. Create a new Strategy enum

    2. Create an inference function of type Callable[[Path, list[str], Any].
        2a. Take a look at 'LLaVA_amount_and_name' or 'textract_micr' as examples.

    3. Add a case statement in the module script under 'match args.strategy:'
"""

import argparse
import boto3
import cv2
import csv
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from tqdm import tqdm
from extract import extract_data
from extract_handwriting import (
    generate_LLaVA_model,
    parse_handwriting,
    ExtractMode,
)
from classify_treasury import is_treasury_check
from extract_micr import extract_micr_data, MICRData, MICRExtractionError
import numpy as np
from typing import Callable, Any

AWS_PROFILE_NAME = 'thwang'
AWS_REGION_NAME = 'us-west-2'

class Strategy(Enum):
    """ Represents an extraction strategy.

    An extraction strategy is any algorithm/method to extract data from a check image. 
    
    These could vary between what they extract (MICR, amount, etc) or what underlying model
    they use to extract them (LLaVA, docTR, Textract, etc).
    """
    LLAVA_AMOUNT_AND_NAME = "LLAVA_AMOUNT_AND_NAME"
    TEXTRACT_MICR = "TEXTRACT_MICR"
    LLAVA_TREASURY = "LLAVA_TREASURY"


def LLaVA_amount_and_name(
        file_path: Path, 
        headers: list[str], 
        model: Any
) -> list[Any]:
    """ Inference function to output check data in the form of [<Check Amount>, <Payer First Name>, <Payer Last Name>].

    Args:
        file_path: File path to input image
        headers: should be ["Check Amount", "Payer First Name", "Payer Last Name"]
        model: A llava model

    The prompt returns "Check Amount : <amount> \n Payer First Name : <first name> ...". To parse this, the string is 
    split into individual tokens based on \n and :. Then, tokens are processed sequentially in pairs. The first token should be in
    headers, and then the second token is put in the correct row. If there are any errors, "NA" will be returned for 
    that value.
    """
    PROMPT = "Scan the check and list only the following information in key:value form separated by newlines: Check Amount, Payer First Name, Payer Last Name. For each piece of information not present in the check, return \"NA\" as the value. The Payer Name is located in printed text at the top left corner of the check. DO NOT use the Payee name which is handwritten in the center of the check. Validate the Check Amount by comparing the handwritten amount with the digits on the right side of the check. "
    headers = ["Check Amount", "Payer First Name", "Payer Last Name"]
    output = parse_handwriting(file_path, None, ExtractMode.LLAVA, model, PROMPT)
    row = ["NA"]*len(headers)
    parts = [part for segment in output.split("\n") for part in segment.split(": ")]
    while(len(parts) > 0):
        label = parts.pop(0)
        if label in headers:
            if len(parts) > 0 and parts[0] not in headers:
                row[headers.index(label)] = parts.pop(0)
    return row

def textract_micr(
        image_path: Path, 
        headers: list[str], 
        textract_client: Any
) -> list[Any]:
    """ Inference function to output the MICR data to a row of [<Check number>, <Payer Account Number>, <Payer Routing Number>].

    Args:
        image_path: File path to input image
        headers: should be ["Check Number", "Payer Account Number", "Payer Routing Number"]
        textract_client: An AWS boto3 textract_client
    """
    try:
        micr_data: MICRData = extract_micr_data(image_path, textract_client)
        micr_output = [micr_data.check_number, micr_data.account_number, micr_data.routing_number]

    except MICRExtractionError:
        return ["NA", "NA", "NA"]

    # in the given labels, leading zeros are removed.
    # This is outside the try catch as we want to see the exception if we are not able to cast the output to an int.
    return [int(number) for number in micr_output]

def LLAVA_treasury(
        file_path: Path, 
        headers: list[str], 
        model: Any
):
    """ Inference function to output whether or not a check is a treasury check, outputting ["TreasuryCheck"], ["Check"]
    or ["NA"] in the case of an error.

    Args:
        file_path: File path to input image
        headers: should be ["Check Type"]
        model: A llava model
    """
    try:
        if is_treasury_check(file_path, model):
            return ["TreasuryCheck"]
        else:
            return ["Check"]
    except ValueError:
        return ["NA"]

# processes check images
def analyze_checks(
        dataset_path: Path, 
        out_file: Path, 
        inference_function: Callable[[Path, list[str], Any], list[Any]],
        headers: list[str]
) -> int:
    """ Function that applies the inference_function to every file in the dataset_path and writes each output
    as a csv row into the out_file. The headers determine the contents of each row / the columns of the overall file.

    Args:
        dataset_path: Folder path to all images meant to be processed.
        out_file: csv file path to write inference output to.
        inference_function: Each strategy has an inference function associated with it. This function takes in a file path
                            and some other parameters and returns a row to write to the csv.
        headers: the column names of the csv.

    This function iterates over all files, in order of their check number. The check number is given by the ## in
    mcd-test-N-front-##.jpg, and the code below depends on files being named in that exact format; only ## can be a
    variable amount of characters. This special ordering (which is not the alphanumeric order of the check names)
    is required as it is the ordering given by the label file from @jts.

    The inference function is then applied to each check in this order, and the output is written to the out_file.
    Whenever an error occurs in check reading or the data is not found, the special string "NA" is expected. This
    is important in the downstream script compare_predictions_to_labels.py
    """
    with open(out_file, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the headers to the CSV file
        csv_writer.writerow(headers)

        # Finicky, relies on files being in the mcd-test-N-front-##.jpg 
        # (e.g. must have exactly 17 chars before the number.)
        def comparator(file_string):
            try:
                return int(file_string[17:-4])
            except:
                # arbitrary large number to kick weird files to the end.
                return 100000000

        files = os.listdir(dataset_path)

        sorted_files = sorted(files, key=comparator)

        # Running the inference function for every file in the input directory.
        for file_name in tqdm(sorted_files, desc="Analyzing check images ..."):
            file_path = os.path.join(dataset_path, file_name)
            file_path = Path(file_path)

            if not os.path.isfile(file_path):
                continue

            row = inference_function(file_path, headers, model)
            csv_writer.writerow(row)

        return len(sorted_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder_path', type=str, help='Input folder of check images.')
    parser.add_argument('out_file', type=str, help='Output .csv file.')
    parser.add_argument('strategy', type=Strategy, choices=Strategy, help='Specify a strategy.')
    parser.add_argument('--hourly_cost', default=5.672, type=float, help='Optional argument')


    args = parser.parse_args()
    data_folder_path = Path(args.data_folder_path)
    outfile_path = Path(args.out_file)

    # Hourly cost for a ml.g5.12xLarge EC2 instance
    HOURLY_COST = args.hourly_cost

    model: Any  # Strategy model
    match args.strategy:
        case Strategy.LLAVA_AMOUNT_AND_NAME:
            headers = ["Check Amount", "Payer First Name", "Payer Last Name"]
            inference_function = LLaVA_amount_and_name
            model = generate_LLaVA_model()

        case Strategy.TEXTRACT_MICR:
            session = boto3.Session(profile_name=AWS_PROFILE_NAME)
            textract_client = session.client('textract', region_name=AWS_REGION_NAME)

            headers = ["Check Number", "Payer Account Number", "Payer Routing Number"]
            inference_function = textract_micr
            model = textract_client

        case Strategy.LLAVA_TREASURY:
            headers = ["Check Type"]
            inference_function = LLAVA_treasury
            model = generate_LLaVA_model()

        case _:
            raise ValueError(f"Invalid strategy: {args.strategy}")

    start_time = datetime.now()

    num_checks_processed = analyze_checks(data_folder_path, outfile_path, inference_function, headers)

    current_time = datetime.now()
    elapsed_time = current_time-start_time
    print("Elapsed Time: ", elapsed_time)
    print("Average Seconds per Inference: ", elapsed_time.total_seconds() / num_checks_processed)
    seconds_per_inference = elapsed_time.total_seconds() / num_checks_processed

    if args.strategy in [Strategy.LLAVA_AMOUNT_AND_NAME, Strategy.LLAVA_TREASURY]:
        print("Cost per Inference in USD: ", seconds_per_inference/(60*60) * HOURLY_COST)

    print(f"Writing results to {outfile_path}")