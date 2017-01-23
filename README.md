# BoxCars Fine-Grained Recognition of Vehicles


## Installation

* Clone the repository and cd to it
```bash
git clone addr BoxCars
cd BoxCars
```
* Create virtual environment for this project - you can use virtualenvwrapper or following commands
* **IMPORTANT NOTE:** this project is using **Python3**
```bash
virtuenv -p /usr/bin/python3 boxcars_venv
source boxcars_venv/bin/activate
```
* Install required packages
```bash
pip install -r requirements.txt
```
* Manually download dataset or use script `python dataset/download.py` (use -h for help)
* (Optional) Download trained models using `python models/download.py` (use -h for help)

## Usage


## Trained models

Net | Single Accuracy | Track Accuracy | Image Processing Time
----|-----------------|----------------|----------------------
ResNet50 |  83.13% |  90.72% | 5.8ms
VGG16 | 83.84% | 92.23% | 5.4ms
VGG19 | 83.91% |  91.99% | 5.4ms


## Links

## Citation