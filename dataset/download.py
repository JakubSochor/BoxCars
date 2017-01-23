# -*- coding: utf-8 -*-
import os
import urllib.request 
import re
import argparse
import sys

#%%
DATASET_URL = "TBD"
DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.realpath(__file__))


#%%
parser = argparse.ArgumentParser(description="Download BoxCars116k dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="output directory where the dataset should be extracted")
args = parser.parse_args()


assert False, "IMPLEMENT"