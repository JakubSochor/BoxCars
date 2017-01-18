# -*- coding: utf-8 -*-
import pickle
import os
from config import OUTPUT_FINAL_WEIGHTS
import numpy as np

#%%
def load_cache(path, encoding="latin-1", fix_imports=True):
    """
    encoding latin-1 is default for Python2 compatibility
    """
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding, fix_imports=True)

#%%
def save_cache(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

#%%
def ensure_dir(d):
    if len(d)  == 0: # for empty dirs (for compatibility with os.path.dirname("xxx.yy"))
        return
    if not os.path.exists(d):
        try:
            os.makedirs(d)
        except OSError as e:
            if e.errno != 17: # FILE EXISTS
                raise e

#%%
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="BoxCars fine-grained recognition algorithm Keras re-implementation",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval", type=str, default=None, help="path to model file to be evaluated")
    parser.add_argument("--resume", type=str, default=None, help="path to model file to be resumed")
    
    args = parser.parse_args()
    assert args.eval is None or args.resume is None, "--eval and --resume are mutually exclusive"
    return args

 