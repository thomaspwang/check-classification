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
from extract_bboxes import textdump_from_path
from extract_handwriting import (
    generate_LLaVA_model,
    parse_handwriting,
    ExtractMode,
)
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
    # TODO: Module Docstring
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
        textract_client: An AWS boto3 textract_client.
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
    PROMPT = "Is the word \"UNITED STATES TREASURY\" written in a Gothic / Old English font present on this check? Only answer one word: True or False."
    output = parse_handwriting(Path(file_path), None, ExtractMode.LLAVA, model, PROMPT)
    if output.upper() == "TRUE":
        return ["TreasuryCheck"]
    elif output.upper() == "FALSE":
        return ["Check"]
    else:
        print("check processing error")
        return ["NA"]

# processes check images
def analyze_checks(
        dataset_path: Path, 
        out_file: Path, 
        inference_function: Callable[[Path, list[str], Any], list[Any]],
        headers: list[str]
) -> int:
    """ TODO: Needs really good module docstring here since this might be confusing
    
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

        # TODO: All this os stuff should be replaced with pathlib Path operations
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