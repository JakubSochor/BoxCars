# -*- coding: utf-8 -*-
import os

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator
from config import *
from utils import ensure_dir, parse_args

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard


#%% initialize dataset
args = parse_args()
dataset = BoxCarsDataset(load_split="hard", load_atlas=True)

#%% get optional path to load model
model = None
for path in [args.eval, args.resume]:
    if path is not None:
        print("Loading model from %s"%path)
        model = load_model(path)
        break

#%% construct the model as it was not passed as an argument
if model is None:
    print("Initializing new model...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(dataset.get_number_of_classes(), activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    optimizer = SGD(lr=LEARNING_RATE, decay=1e-4, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

#%% training
if args.eval is None:
    print("Training...")
    #%% initialize dataset for training
    dataset.initialize_data("train")
    dataset.initialize_data("validation")
    generator_train = BoxCarsDataGenerator(dataset, "train", BATCH_SIZE, training_mode=True)
    generator_val = BoxCarsDataGenerator(dataset, "validation", BATCH_SIZE, training_mode=False)


    #%% callbacks
    ensure_dir(OUTPUT_TENSORBOARD_DIR)
    ensure_dir(OUTPUT_SNAPSHOTS_DIR)
    tb_callback = TensorBoard(OUTPUT_TENSORBOARD_DIR, histogram_freq=1, write_graph=False, write_images=False)
    saver_callback = ModelCheckpoint(os.path.join(OUTPUT_SNAPSHOTS_DIR, "model_{epoch:03d}_{val_acc:.2f}.h5"), period=4 )

    #%% get initial epoch
    initial_epoch = 0
    if args.resume is not None:
        initial_epoch = int(os.path.basename(args.resume).split("_")[1]) + 1


    model.fit_generator(generator=generator_train, 
                        samples_per_epoch=generator_train.N,
                        nb_epoch=20,
                        verbose=1,
                        validation_data=generator_val,
                        nb_val_samples=generator_val.N,
                        callbacks=[tb_callback, saver_callback],
                        initial_epoch = initial_epoch,
                        )

    #%% save trained data
    print("Saving the final model to %s"%(OUTPUT_FINAL_MODEL))
    ensure_dir(os.path.dirname(OUTPUT_FINAL_MODEL))
    model.save(OUTPUT_FINAL_MODEL)


#%% evaluate the model 
print("Running evaluation...")
dataset.initialize_data("test")
generator_test = BoxCarsDataGenerator(dataset, "test", BATCH_SIZE, training_mode=False, generate_y=False)
predictions = model.predict_generator(generator_test, generator_test.N)
single_acc, tracks_acc = dataset.evaluate(predictions)
print("Accuracy: %.2f%%"%(single_acc*100))
print("Track accuracy: %.2f%%"%(tracks_acc*100))
