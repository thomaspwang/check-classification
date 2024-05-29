"""
Extraction Package

This package provides a set of modules and functions to perform various tasks related to extracting information from check images.
It supports different OCR models including LLaVA and DocTR, and provides tools for handling bounding boxes, extracting MICR data,
classifying checks, and more.

Modules:
    analyze_checks:
        Script for analyzing a folder of checks and extracting data based on specified strategies to a CSV file. 
        It supports various extraction strategies like LLaVA and Textract.

    extract_bboxes:
        Module for extracting and visualizing bounding boxes from check images using AWS Textract.

    classify_treasury:
        Module for determining whether a check is a United States Treasury check using the LLaVA model.

    parse_bbox:
        Module for extracting specific text from images using various OCR models, such as LLaVA and DocTR.
        It handles the extraction process by cropping the image to the bounding box and passing it to the appropriate model.

    extract:
        Module for extracting all text data from a check image using either LLaVA or DocTR models.

    extract_micr:
        Module for extracting MICR (Magnetic Ink Character Recognition) data from check images using AWS Textract.

    extraction_utils:
        Utility functions for check data extraction. Provides functions for cropping images to bounding boxes, 
        merging bounding boxes, and stretching bounding boxes.

    llava_wrapper:
        Wraps the LLaVA library into a callable class, which allows one to feed in a prompt and an image
        and recieve a text response.
"""