"""Allows the package to be installed with pip install -e.

We do this so that we can import our scripts as modules for testing without
altering sys.path. No need to run this file directly, just run:

    pip install -e

as per the setup instructions in the README.
"""
from setuptools import setup, find_packages

setup(
    name="sofi-check-classification",
    version="0.1",
    packages=find_packages(),
)