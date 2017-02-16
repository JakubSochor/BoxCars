#!/bin/bash

for net in ResNet50 VGG16 VGG19 InceptionV3
do
	python scripts/train_eval.py --train-net $net
done