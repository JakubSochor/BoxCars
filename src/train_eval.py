# -*- coding: utf-8 -*-
import os

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator
from config import *
from utils import ensure_dir, parse_args

from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard


#%% initialize dataset
args = parse_args()
dataset = BoxCarsDataset(load_split="hard", load_atlas=True)

#%% construct the model (this can be changed to different net)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base_model.output
x = Flatten()(x)
predictions = Dense(dataset.get_number_of_classes(), activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

#%% compile the model
optimizer = SGD(lr=LEARNING_RATE, decay=1e-4, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])


if args.eval is None:
    print("Training...")
    #%% initialize dataset for training
    dataset.initialize_data("train")
    dataset.initialize_data("validation")
    generator_train = BoxCarsDataGenerator(dataset, "train", BATCH_SIZE, training_mode=True)
    generator_val = BoxCarsDataGenerator(dataset, "validation", BATCH_SIZE, training_mode=False)


    #%% train
    ensure_dir(OUTPUT_TENSORBOARD_DIR)
    ensure_dir(OUTPUT_SNAPSHOTS_DIR)
    tb_callback = TensorBoard(OUTPUT_TENSORBOARD_DIR, histogram_freq=1, write_graph=False, write_images=False)
    saver_callback = ModelCheckpoint(os.path.join(OUTPUT_SNAPSHOTS_DIR, "weights.{epoch:02d}-{val_acc:.2f}.h5") )

    model.fit_generator(generator=generator_train, 
                        samples_per_epoch=generator_train.N,
                        nb_epoch=15,
                        verbose=1,
                        validation_data=generator_val,
                        nb_val_samples=generator_val.N,
                        callbacks=[tb_callback, saver_callback],
                        )

    #%% save trained data
    print("Saving the final model to %s"%(OUTPUT_FINAL_MODEL))
    ensure_dir(os.path.dirname(OUTPUT_FINAL_MODEL))
    model.save(OUTPUT_FINAL_MODEL)
    print("Saving the final weights to %s"%(OUTPUT_FINAL_WEIGHTS))
    ensure_dir(os.path.dirname(OUTPUT_FINAL_WEIGHTS))
    model.save_weights(OUTPUT_FINAL_WEIGHTS)

else:
    print("Loading weights from %s"%args.eval)
    model.load_weights(args.eval)

#%% evaluate the model (accuracy only for one sample - per track accuracy needs to be implemented)
print("Running evaluation...")
dataset.initialize_data("test")
generator_test = BoxCarsDataGenerator(dataset, "test", BATCH_SIZE, training_mode=False)
eval_results = model.evaluate_generator(generator_test, generator_test.N)
print("Evaluation done.")
for metric_name, result in zip(model.metrics_names, eval_results):
    print("%s: %.3f"%(metric_name, result))
