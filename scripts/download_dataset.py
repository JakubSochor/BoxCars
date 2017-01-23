# -*- coding: utf-8 -*-
import _init_paths
import os
import urllib.request 
import re
import argparse
import sys
from utils import ensure_dir, download_report_hook
import zipfile

#%%
DATASET_URL = "https://medusa.fit.vutbr.cz/traffic/data/BoxCars116k.zip"
DEFAULT_OUTPUT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
SAVE_ZIP_NAME = "BoxCars116k.zip"


#%%
parser = argparse.ArgumentParser(description="Download BoxCars116k dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-dir","-o", type=str, default=DEFAULT_OUTPUT_DIR, help="output directory for the dataset")
args = parser.parse_args()

#%%
ensure_dir(args.output_dir)
os.chdir(args.output_dir)

#%%
print("Downloading BoxCars116k dataset to to %s... "%(args.output_dir), end="")
sys.stdout.flush()
urllib.request.urlretrieve(DATASET_URL, SAVE_ZIP_NAME, download_report_hook)

#%%
print("Extracting contents... ", end="")
with zipfile.ZipFile(SAVE_ZIP_NAME, "r") as archive:
    members = archive.infolist()
    for i, member in enumerate(members):
        percents = i/len(members) * 100
        show_str = " %.1f%%"%(percents)
        sys.stdout.write(show_str + len(show_str)*"\b")
        sys.stdout.flush()
        archive.extract(member)
    show_str = " %.1f%%"%(100)
    print(show_str)
    
