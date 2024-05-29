#!./venv/bin/python3.10
"""Converts .numbers file to .csv.

This script takes a .numbers file as input and converts it to a .csv file. 

Usage:
    python convert_numbers_to_csv.py file_path output_path

Arguments:
    file_path (str): Path to the .numbers file to be converted.
    output_path (str): Path where the resulting .csv file will be saved.

Example:
    python convert_numbers_to_csv.py input.numbers output.csv
"""
import argparse

from numbers_parser import Document
from pathlib import Path
import pandas as pd


def convert_numbers_to_csv(file_path: Path, output_path: Path):
    # Load the .numbers document
    doc = Document(file_path)

    # Access the sheets attribute directly
    sheets = doc.sheets

    # Extract data from the first sheet and the first table
    sheet = sheets[0]
    table = sheet.tables[0]

    # Convert the table to a Pandas DataFrame
    data = []
    for row in table.rows():
        data.append([cell.value for cell in row])

    df = pd.DataFrame(data)

    # Save the DataFrame to a .csv file
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='TODO')
    parser.add_argument('output_path', type=str, help='TODO')
    args = parser.parse_args()
    
    file_path: Path = Path(args.file_path)
    output_path: Path = Path(args.output_path)

    print(f"Converting {file_path} to a csv file ...")
    convert_numbers_to_csv(file_path, output_path)
    print(f"Done. Output: {output_path}")
