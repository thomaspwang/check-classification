#!./venv/bin/python3.10
""" Script for benchmarking extraction algorithms.
"""

import argparse
import boto3
import cv2
import csv
import os
from datetime import datetime
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
PROMPT = "Scan the check and list only the following information in key:value form separated by newlines: Check Number, Check Amount, Payer First Name, Payer Last Name, Payer Routing Number, Payer Account Number. For your information, the Check Number is typically less than ten thousand and is at the top right of the check. The Routing Number is at the bottom left of the check and is at least nine digits long, and afterwards is the Payer Account Number which is also at least nine digits long. Supplement your response with text extracted by an OCR engine from the check: "

# processes check images
def processCheck(dataset_path, labels, out_file) -> int:
    headers = ["Check Number", "Check Amount", "Payer First Name", "Payer Last Name", "Payer Routing Number", "Payer Account Number"]

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
            textdump = textdump_from_path(Path(file_path), textract_client)
            out = parse_handwriting(Path(file_path), None, ExtractMode.LLAVA, model, PROMPT + " ".join(textdump))
            if (counter %100 == 3 or counter %100 == 4):
                print(out)
            row = ["NA","NA","NA","NA","NA","NA"]
            parts = [part for segment in out.split("\n") for part in segment.split(": ")]
            #print(parts)
            while(len(parts) > 0):
                label = parts.pop(0)
                print(label)
                if label == "Check Number":
                    if len(parts) > 0 and parts[0] not in headers:
                        row[0] = parts.pop(0)
                elif label == "Check Amount":
                    if len(parts) > 0 and parts[0] not in headers:
                        row[1] = parts.pop(0)
                        row[1] = row[1].replace("$", "")

                elif label == "Payer First Name":
                    if len(parts) > 0 and parts[0] not in headers:
                        row[2] = parts.pop(0)
                elif label == "Payer Last Name":
                    if len(parts) > 0 and parts[0] not in headers:
                        row[3] = parts.pop(0)
                elif label == "Payer Routing Number":
                    if len(parts) > 0 and parts[0] not in headers:
                        row[4] = parts.pop(0)
                elif label == "Payer Account Number":
                    if len(parts) > 0 and parts[0] not in headers:
                        row[5] = parts.pop(0)
            csv_writer.writerow(row)
        return counter



if __name__ == "__main__":
    # TODO: Improve cmd interface

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='TODO')
    parser.add_argument('labels', type=str, help='TODO')
    parser.add_argument('out_file', type=str, help='TODO')

    args = parser.parse_args()

    start_time = datetime.now()
    print("Current time:", start_time)
    numChecksProcessed = processCheck(args.dataset_path, args.labels, args.out_file)
    current_time = datetime.now()
    elapsed_time = current_time-start_time
    print("Elapsed time: ", elapsed_time)
    print("average seconds per inference: ", elapsed_time.total_seconds() / numChecksProcessed)
    seconds_per_inference = elapsed_time.total_seconds() / numChecksProcessed
    print("cost per inference in dollars: ", seconds_per_inference/(60*60) * 5.672)