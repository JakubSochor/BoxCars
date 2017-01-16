# -*- coding: utf-8 -*-
from config import BOXCARS_DATASET_ROOT
from utils import load_cache
import os
import cv2
#%%
class BoxCarsDataset(object):
    def __init__(self, load_atlas = False, load_split = None, use_estimated_3DBB = False, estimated_3DBB_path = None):
        self.dataset = load_cache(os.path.join(BOXCARS_DATASET_ROOT, "dataset.pkl"))
        self.use_estimated_3DBB = use_estimated_3DBB
        
        self.atlas = None
        self.split = None
        self.split_name = None
        self.estimated_3DBB = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None
        if load_atlas:
            self.load_atlas()
        if load_split is not None:
            self.load_classification_split(load_split)
        if self.use_estimated_3DBB:
            self.estimated_3DBB = load_cache(estimated_3DBB_path)
        
    #%%
    def load_atlas(self):
        self.atlas = load_cache(os.path.join(BOXCARS_DATASET_ROOT, "atlas.pkl"))
    
    #%%
    def load_classification_split(self, split_name):
        self.split = load_cache(os.path.join(BOXCARS_DATASET_ROOT, "classification_splits.pkl"))[split_name]
        self.split_name = split_name
       
    #%%
    def get_image(self, vehicle_id, instance_id):
        return cv2.imdecode(self.atlas[vehicle_id][instance_id], 1)
        
    #%%
    def get_3DBB(self, vehicle_id, instance_id, original_image_coordinates=False):
        if not self.use_estimated_3DBB:
            bb3d = self.dataset["samples"][vehicle_id]["instances"][instance_id]["3DBB"]
        else:
            bb3d = self.estimated_3DBB[vehicle_id][instance_id]
            
        if original_image_coordinates:
            return bb3d
        else:
            return bb3d - self.dataset["samples"][vehicle_id]["instances"][instance_id]["3DBB_offset"]
            
       
    #%%
    def initialize_data(self, train=False, val=False, test=False):
        if train:
            self.X_train, self.Y_train = self._initialize_data("train")
        if val:
            self.X_val, self.Y_val = self._initialize_data("validation")
        if test:
            self.X_test, self.Y_test = self._initialize_data("test")
    
    #%%
    def _initialize_data(self, part):
        data = self.split[part]
        x, y = [], []
        for vehicle_id, label in data:
            num_instances = len(self.dataset["samples"][vehicle_id]["instances"])
            x.extend([(vehicle_id, instance_id) for instance_id in range(num_instances)])
            y.extend([label]*num_instances)
        return x,y
        
        
            
        
            
        
        
        
