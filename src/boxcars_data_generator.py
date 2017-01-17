# -*- coding: utf-8 -*-
import cv2
import numpy as np
from keras.preprocessing.image import Iterator
from boxcars_image_transformations import alter_HSV, image_drop, unpack_3DBB, add_bb_noise_flip
import random

#%%
class BoxCarsDataGenerator(Iterator):
    def __init__(self, dataset, part, batch_size=8, training_mode=False, seed=None, generate_y = True, image_size = (224,224)):
        assert image_size == (224,224), "only images 224x224 are supported by unpack_3DBB for now, if necessary it can be changed"
        assert dataset.X[part] is not None, "load some classification split first"
        super().__init__(dataset.X[part].shape[0], batch_size, training_mode, seed)
        self.part = part
        self.generate_y = generate_y
        self.dataset = dataset
        self.image_size = image_size
        self.training_mode = training_mode
        if self.dataset.atlas is None:
            self.dataset.load_atlas()

    #%%
    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        x = np.empty([current_batch_size] + list(self.image_size) + [3], dtype=np.float32)
        for i, ind in enumerate(index_array):
            vehicle_id, instance_id = self.dataset.X[self.part][ind]
            vehicle, instance, bb3d = self.dataset.get_vehicle_instance_data(vehicle_id, instance_id)
            image = self.dataset.get_image(vehicle_id, instance_id)
            if self.training_mode:
                image = alter_HSV(image) # randomly alternate color
                image = image_drop(image) # randomly remove part of the image
                bb_noise = np.clip(np.random.randn(2) * 1.5, -5, 5) # generate random bounding box movement
                flip = bool(random.getrandbits(1)) # random flip
                image, bb3d = add_bb_noise_flip(image, bb3d, flip, bb_noise) 
            image = unpack_3DBB(image, bb3d) 
            image = (image.astype(np.float32) - 116)/128.
            x[i, ...] = image
        if not self.generate_y:
            return x
        y = self.dataset.Y[self.part][index_array]
        return x, y

