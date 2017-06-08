# -*- coding: utf-8 -*-
import pickle
import os
import numpy as np
import sys

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
def parse_args(available_nets):
    import argparse
    default_cache = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "cache"))
    parser = argparse.ArgumentParser(description="BoxCars fine-grained recognition algorithm Keras re-implementation",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval", type=str, default=None, help="path to model file to be evaluated")
    parser.add_argument("--resume", type=str, default=None, help="path to model file to be resumed")
    parser.add_argument("--train-net", type=str, default=available_nets[0], help="train on one of following nets: %s"%(str(available_nets)))
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0025, help="learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="run for epochs")
    parser.add_argument("--cache", type=str, default=default_cache, help="where to store training meta-data and final model")
    parser.add_argument("--estimated-3DBB", type=str, default=None, help="use estimated 3DBBs from specified path")
    
    
    args = parser.parse_args()
    assert args.eval is None or args.resume is None, "--eval and --resume are mutually exclusive"
    if args.eval is None and args.resume is None:
        assert args.train_net in available_nets, "--train-net must be one of %s"%(str(available_nets))

    return args

 
#%%
def download_report_hook(block_num, block_size, total_size):
    downloaded = block_num*block_size
    percents = downloaded / total_size * 100
    show_str = " %.1f%%"%(percents)
    sys.stdout.write(show_str + len(show_str)*"\b")
    sys.stdout.flush()
    if downloaded >= total_size:
        print()
