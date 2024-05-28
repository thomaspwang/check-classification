#!./venv/bin/python3.10
""" Script for benchmarking extraction algorithms.
"""

import argparse
import boto3
import cv2
import csv
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from extract import extract_data
from extract_bboxes import textdump_from_path
from extract_handwriting import (
    generate_LLaVA_model,
    parse_handwriting,
    ExtractMode,
)
import numpy as np


AWS_PROFILE_NAME = 'thwang'
AWS_REGION_NAME = 'us-west-2'

session = boto3.Session(profile_name=AWS_PROFILE_NAME)
textract_client = session.client('textract', region_name=AWS_REGION_NAME)

model = generate_LLaVA_model()

def LLAVA_amount_and_name(file_path, headers):
    PROMPT = "Scan the check and list only the following information in key:value form separated by newlines: Check Amount, Payer First Name, Payer Last Name. For each piece of information not present in the check, return \"NA\" as the value. The Payer Name is located in printed text at the top left corner of the check. DO NOT use the Payee name which is handwritten in the center of the check. Validate the Check Amount by comparing the handwritten amount with the digits on the right side of the check. "
    headers = ["Check Amount", "Payer First Name", "Payer Last Name"]
    output = parse_handwriting(Path(file_path), None, ExtractMode.LLAVA, model, PROMPT)
    row = ["NA"]*len(headers)
    parts = [part for segment in output.split("\n") for part in segment.split(": ")]
    while(len(parts) > 0):
        label = parts.pop(0)
        if label in headers:
            if len(parts) > 0 and parts[0] not in headers:
                row[headers.index(label)] = parts.pop(0)
    return row

def LLAVA_treasury(file_path, headers):
    PROMPT = "Is the word \"UNITED STATES TREASURY\" written in a Gothic / Old English font present on this check? Only answer one word: True or False."
    output = parse_handwriting(Path(file_path), None, ExtractMode.LLAVA, model, PROMPT)
    if output.upper() == "TRUE":
        return ["TreasuryCheck"]
    elif output.upper() == "FALSE":
        return ["Check"]
    else:
        print("check processing error")
        return ["NA"]

class Strategy(Enum):
    LLAVA_amount_and_name = "LLAVA_amount_and_name"
    LLAVA_treasury = "LLAVA_treasury"

# processes check images
def analyzeChecks(dataset_path, out_file, strategy_to_eval, headers) -> int:

    with open(out_file, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the headers to the CSV file
        csv_writer.writerow(headers)
        counter = 0

        # Finicky, relies on files being in the mcd-test-N-front-##.jpg 
        # (e.g. must have exactly 17 chars before the number.)
        def comparator(file_string):
            try:
                return int(file_string[17:-4])
            except:
                # arbitrary large number to kick weird files to the end.
                return 100000000

        files = os.listdir(dataset_path)
        for file_name in sorted(files, key=comparator):
            file_path = os.path.join(dataset_path, file_name)

            if not os.path.isfile(file_path):
                continue

            counter += 1
            if counter % 50 == 0:
                print(counter)
            # todo: add nice tqdm progress bar

            row = strategy_to_eval(file_path, headers)
            csv_writer.writerow(row)
        return counter


if __name__ == "__main__":
    # TODO: Improve cmd interface

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='TODO')
    parser.add_argument('out_file', type=str, help='TODO')
    parser.add_argument('strategy', type=Strategy, choices=Strategy, help='Choose a strategy')

    args = parser.parse_args()

    match args.strategy:
        case Strategy.LLAVA_amount_and_name:
            headers = ["Check Amount", "Payer First Name", "Payer Last Name"]
            fn_to_eval = LLAVA_amount_and_name
        case Strategy.LLAVA_treasury:
            headers = ["Check Type"]
            fn_to_eval = LLAVA_treasury
        case _:
            raise ValueError(f"Invalid strategy: {args.strategy}")

    start_time = datetime.now()
    print("Current time:", start_time)
    numChecksProcessed = analyzeChecks(args.dataset_path, args.out_file, fn_to_eval, headers)
    current_time = datetime.now()
    elapsed_time = current_time-start_time
    print("Elapsed time: ", elapsed_time)
    print("average seconds per inference: ", elapsed_time.total_seconds() / numChecksProcessed)
    seconds_per_inference = elapsed_time.total_seconds() / numChecksProcessed
    print("cost per inference in dollars: ", seconds_per_inference/(60*60) * 5.672)