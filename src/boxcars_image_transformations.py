# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random


#%%
def alter_HSV(img, change_probability = 0.6):
    if random.random() < 1-change_probability:
        return img
    addToHue = random.randint(0,179)
    addToSaturation = random.gauss(60, 20)
    addToValue = random.randint(-50,50)
    hsvVersion =  cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    channels = hsvVersion.transpose(2, 0, 1)
    channels[0] = ((channels[0].astype(int) + addToHue)%180).astype(np.uint8)
    channels[1] = (np.maximum(0, np.minimum(255, (channels[1].astype(int) + addToSaturation)))).astype(np.uint8)
    channels[2] = (np.maximum(0, np.minimum(255, (channels[2].astype(int) + addToValue)))).astype(np.uint8)
    hsvVersion = channels.transpose(1,2,0)   
        
    return cv2.cvtColor(hsvVersion, cv2.COLOR_HSV2RGB)

#%%
def image_drop(img, change_probability = 0.6):
    if random.random() < 1-change_probability:
        return img
    width = random.randint(int(img.shape[1]*0.10), int(img.shape[1]*0.3))
    height = random.randint(int(img.shape[0]*0.10), int(img.shape[0]*0.3))
    x = random.randint(int(img.shape[1]*0.10), img.shape[1]-width-int(img.shape[1]*0.10))
    y = random.randint(int(img.shape[0]*0.10), img.shape[0]-height-int(img.shape[0]*0.10))
    img[y:y+height,x:x+width,:] = (np.random.rand(height,width,3)*255).astype(np.uint8)
    return img

#%%
def add_bb_noise_flip(image, bb3d, flip, bb_noise):
    bb3d = bb3d + bb_noise 
    if flip:
        bb3d[:, 0] = image.shape[1] - bb3d[:,0]
        image = cv2.flip(image, 1)
    return image, bb3d

#%%
def _unpack_side(img, origPoints, targetSize):
    origPoints = np.array(origPoints).reshape(-1,1,2)
    targetPoints = np.array([(0,0), (targetSize[0],0), (0, targetSize[1]), 
                             (targetSize[0], targetSize[1])]).reshape(-1,1,2).astype(origPoints.dtype)
    m, _ = cv2.findHomography(origPoints, targetPoints, 0)
    resultImage = cv2.warpPerspective(img, m, targetSize)
    return resultImage
    
    
#%%    
def unpack_3DBB(img, bb):
    frontal = _unpack_side(img, [bb[0], bb[1], bb[4], bb[5]], (75,124))
    side = _unpack_side(img, [bb[1], bb[2], bb[5], bb[6]], (149,124))
    roof = _unpack_side(img, [bb[0], bb[3], bb[1], bb[2]], (149,100))
    
    final = np.zeros((224,224,3), dtype=frontal.dtype)
    final[100:, 0:75] = frontal
    final[0:100, 75:] = roof
    final[100:, 75:] = side
    
    return final
    