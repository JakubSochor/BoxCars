# -*- coding: utf-8 -*-
import pickle

def load_cache(path, encoding="latin-1", fix_imports=True):
    with open(path, "rb") as f:
        return pickle.load(f, encoding=encoding, fix_imports=True)
