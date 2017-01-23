# -*- coding: utf-8 -*-
import _init_paths
import os
import urllib.request 
import re
import argparse
import sys
from utils import ensure_dir, download_report_hook

#%%
MODELS_DIR_URL = "https://medusa.fit.vutbr.cz/traffic/data/BoxCars-models/"
SUFFIX = "h5"
DEFAULT_OUTPUT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "models"))

#%%
with urllib.request.urlopen(MODELS_DIR_URL) as response:
    dir_listing = response.read().decode("utf-8")

model_matcher = re.compile(r'href="(.*)\.%s"'%(SUFFIX))
available_nets = model_matcher.findall(dir_listing)


#%%
parser = argparse.ArgumentParser(description="Download trained model files. Available nets: %s"%(str(available_nets)),
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-dir","-o", type=str, default=DEFAULT_OUTPUT_DIR, help="output directory where to put downloaded models")
parser.add_argument("--all", "-a", default=False, action="store_true", help="download all available models")
parser.add_argument("net_name", nargs="*")
args = parser.parse_args()

download_nets = args.net_name
if args.all:
    download_nets = available_nets
    
if len(download_nets) == 0:
    print("You need to specify nets to download or use --all to download all of them\nAVAILABLE NETS: %s\n"%(str(available_nets)))
    parser.print_usage()
    sys.exit(1)

#%%
print("Saving downloaded models to: %s"%(args.output_dir))
ensure_dir(args.output_dir)
for net in download_nets:
    if net not in available_nets:
        print("WARNING: Skipping %s because it is not available. AVAILABLE_NETS: %s"%(net, str(available_nets)))
        continue
    print("Downloading %s... "%(net), end="")
    sys.stdout.flush()
    urllib.request.urlretrieve(MODELS_DIR_URL + net + "." + SUFFIX, os.path.join(args.output_dir, "%s.%s"%(net, SUFFIX)), download_report_hook)
    
