from _ast import Global

import numpy as np
import argparse
import json
from os import *

from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, MaxPool2D, AveragePooling2D
from keras.layers import Flatten, Dense, Activation, BatchNormalization, Dropout
from keras.models import model_from_json

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

preprocess_input = imagenet_utils.preprocess_input

ap = argparse.ArgumentParser()

ap.add_argument("-m", "--mode", required=True, help="train or predict")
ap.add_argument("-i", "--image", help="path to the input image")
args = vars(ap.parse_args())

mode = args["mode"]

version = "v3"
num_of_classes = 120
target_size = (224, 224)
batch_size = 128

tensorboard = TensorBoard(log_dir="tensor_logs/cnn_" + version)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(AveragePooling2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(AveragePooling2D())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())
model.add(Dropout(0.33))

model.add(Dense(num_of_classes, activation='softmax'))

model.summary()

if mode == "train":

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory('./train/',
                                                        target_size=target_size, color_mode='rgb',
                                                        batch_size=batch_size, class_mode='categorical',
                                                        shuffle=True, seed=42)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    val_generator = val_datagen.flow_from_directory('./validation/',
                                                    target_size=target_size,color_mode="rgb",
                                                    batch_size=batch_size, class_mode="categorical")

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = val_generator.n // val_generator.batch_size

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        validation_data=val_generator,
                        validation_steps=step_size_valid,
                        callbacks=[tensorboard],
                        epochs=100)

    model.save_weights("doggle_cnn_" + version + ".h5") #Save weights
    with open("doggle_cnn_arc_" + version + ".json", 'w') as f: #Save Architecture
        f.write(model.to_json())

    # Save class indices for prediction labels
    np.save("doggle_cnn_classes_" + version + ".txt", train_generator.class_indices)

elif mode == "predict":

    with open("doggle_cnn_arc_" + version + ".json", 'r') as f:
        model = model_from_json(f.read())

    model.load_weights("doggle_cnn_" + version + ".h5")

    # Load class indices for prediction labels
    if os.path.isfile("doggle_cnn_classes_" + version + ".txt"):
        class_indices = np.load("doggle_cnn_classes_" + version + ".txt").item()

    orig_img = image.load_img(args["image"], target_size=target_size)

    img = np.expand_dims(orig_img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)
    results = []
    maxRows = 3
    classes = class_indices

    for pred in preds:

        top_indices = pred.argsort()[-maxRows:][::-1]
        for i in top_indices:
            clsName = list(classes.keys())[list(classes.values()).index(i)]
            result = clsName + ":" + str(pred[i])
            results.append(result)

    for res in results:
        print(res)
