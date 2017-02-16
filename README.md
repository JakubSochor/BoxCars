# BoxCars Fine-Grained Recognition of Vehicles
This is Keras+Tensorflow re-implementation of our method for fine-grained classification of vehicles decribed in **TODO NAME**.
The numerical results are slightly different, but similar. This code is for **research only** purposes.
If you use the code, please cite our paper:
```
**TODO CITATION**
```

## Installation

* Clone the repository and cd to it.

```bash
git clone https://github.com/JakubSochor/BoxCars.git BoxCars
cd BoxCars
```
* (Optional, but recommended) Create virtual environment for this project - you can use **virtualenvwrapper** or following commands. **IMPORTANT NOTE:** this project is using **Python3**.

```bash
virtuenv -p /usr/bin/python3 boxcars_venv
source boxcars_venv/bin/activate
```

* Install required packages 

```bash
pip3 install -r requirements.txt 
```

* Manually download dataset **TODO LINK** and unzip it.
* Modify `scripts/config.py` and change `BOXCARS_DATASET_ROOT` to directory where is the unzipped dataset.
* (Optional) Download trained models using `scripts/download_models.py`. To download all models to default location (`./models`) run following command (or use -h for help):

```base
python3 scripts/download_models.py --all
``` 


## Usage
### Fine-tuning of the Models
To fine-tune a model use `scripts/train_eval.py` (use -h for help). Example for ResNet50:
```bash
python3 scripts/train_eval.py --train-net ResNet50 
```
It is also possible to resume training using `--resume` argument for `train_eval.py`.

### Evaluation
The model is evaluated when the training is finished, however it is possible to evaluate saved model by running:
```bash
python3 scripts/train_eval.py path-to-model.h5
```


## Trained models
We provide numerical results of models distributed with this code (use `scripts/download_models.py`). 
The processing time was measured on GTX1080 with CUDNN.

Net | Single Accuracy | Track Accuracy | Image Processing Time
----|-----------------|----------------|----------------------
ResNet50 |  83.13% |  90.72% | 5.8ms
VGG16 | 83.84% | 92.23% | 5.4ms
VGG19 | 83.91% |  91.99% | 5.4ms


## Links 
* BoxCars116k dataset **TODO LINK**
* Web with our [Traffic Research](https://medusa.fit.vutbr.cz/traffic/)
