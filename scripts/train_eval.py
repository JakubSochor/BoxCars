# -*- coding: utf-8 -*-
import _init_paths
# this should be soon to prevent tensorflow initialization with -h parameter
from utils import ensure_dir, parse_args
args = parse_args(["ResNet50", "VGG16", "VGG19", "InceptionV3"])

# other imports
import os
import time
import sys

from boxcars_dataset import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard


#%% initialize dataset
if args.estimated_3DBB is None:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                             use_estimated_3DBB = True, estimated_3DBB_path = args.estimated_3DBB)

#%% get optional path to load model
model = None
for path in [args.eval, args.resume]:
    if path is not None:
        print("Loading model from %s"%path)
        model = load_model(path)
        break

#%% construct the model as it was not passed as an argument
if model is None:
    print("Initializing new %s model ..."%args.train_net)
    if args.train_net in ("ResNet50", ):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        x = Flatten()(base_model.output)
        
    if args.train_net in ("VGG16", "VGG19"):
        if args.train_net == "VGG16":
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
        elif args.train_net == "VGG19":
            base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
        x = Flatten()(base_model.output)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)

    if args.train_net in ("InceptionV3", ):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
        output_dim = int(base_model.outputs[0].get_shape()[1])
        x = AveragePooling2D((output_dim, output_dim), strides=(output_dim, output_dim), name='avg_pool')(base_model.output)
        x = Flatten()(x)
            
    predictions = Dense(dataset.get_number_of_classes(), activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions, name="%s%s"%(args.train_net, {True: "_estimated3DBB", False:""}[args.estimated_3DBB is not None]))
    optimizer = SGD(lr=args.lr, decay=1e-4, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])


print("Model name: %s"%(model.name))
if args.estimated_3DBB is not None and "estimated3DBB" not in model.name:
    print("ERROR: using estimated 3DBBs with model trained on original 3DBBs")
    sys.exit(1)
if args.estimated_3DBB is None and "estimated3DBB" in model.name:
    print("ERROR: using model trained on estimated 3DBBs and running on original 3DBBs")
    sys.exit(1)

args.output_final_model_path = os.path.join(args.cache, model.name, "final_model.h5")
args.snapshots_dir = os.path.join(args.cache, model.name, "snapshots")
args.tensorboard_dir = os.path.join(args.cache, model.name, "tensorboard")

#%% training
if args.eval is None:
    print("Training...")
    #%% initialize dataset for training
    dataset.initialize_data("train")
    dataset.initialize_data("validation")
    generator_train = BoxCarsDataGenerator(dataset, "train", args.batch_size, training_mode=True)
    generator_val = BoxCarsDataGenerator(dataset, "validation", args.batch_size, training_mode=False)


    #%% callbacks
    ensure_dir(args.tensorboard_dir)
    ensure_dir(args.snapshots_dir)
    tb_callback = TensorBoard(args.tensorboard_dir, histogram_freq=1, write_graph=False, write_images=False)
    saver_callback = ModelCheckpoint(os.path.join(args.snapshots_dir, "model_{epoch:03d}_{val_acc:.2f}.h5"), period=4 )

    #%% get initial epoch
    initial_epoch = 0
    if args.resume is not None:
        initial_epoch = int(os.path.basename(args.resume).split("_")[1]) + 1


    model.fit_generator(generator=generator_train, 
                        samples_per_epoch=generator_train.n,
                        nb_epoch=args.epochs,
                        verbose=1,
                        validation_data=generator_val,
                        nb_val_samples=generator_val.n,
                        callbacks=[tb_callback, saver_callback],
                        initial_epoch = initial_epoch,
                        )

    #%% save trained data
    print("Saving the final model to %s"%(args.output_final_model_path))
    ensure_dir(os.path.dirname(args.output_final_model_path))
    model.save(args.output_final_model_path)


#%% evaluate the model 
print("Running evaluation...")
dataset.initialize_data("test")
generator_test = BoxCarsDataGenerator(dataset, "test", args.batch_size, training_mode=False, generate_y=False)
start_time = time.time()
predictions = model.predict_generator(generator_test, generator_test.n)
end_time = time.time()
single_acc, tracks_acc = dataset.evaluate(predictions)
print(" -- Accuracy: %.2f%%"%(single_acc*100))
print(" -- Track accuracy: %.2f%%"%(tracks_acc*100))
print(" -- Image processing time: %.1fms"%((end_time-start_time)/generator_test.n*1000))
