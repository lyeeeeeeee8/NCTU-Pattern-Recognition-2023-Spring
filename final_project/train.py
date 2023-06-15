#!/usr/bin/env python
# coding: utf-8
import os
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random

from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import util


# Initialization ------------------------------------------------------------------------------------
print("Tensorflow version: ", tf.__version__)       #2.4.0
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
random.seed(seed)
fraction = 0.8
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = fraction
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

train_epoch = 100
batch_size = 16          # Batch size for training and validation
img_width = 100         # Desired image dimensions
img_height = 72         # Desired image dimensions
downsample_factor = 4   # Factor by which the image is going to be downsampled by the convolutional blocks
max_length = 4          # Maximum length of any captcha in the data


# Read training data ------------------------------------------------------------------------------------
dataset_path = "./dataset"
data_dir = Path(f"{dataset_path}/train/")
images = list(data_dir.glob("task*/*.png"))         # Get list of all the images: Posix_Path
print("Number of images: ", len(images))      


dataset = pd.read_csv(f"{dataset_path}/train/annotations.csv")          # Store image-label info
dataset["filepath"] = f"{dataset_path}/train/" + dataset["filename"]    # Append "filepath"
dataset = dataset.sample(frac=1.).reset_index(drop=True)                # Shuffle the dataset
print(dataset.head())

characters = set()                              # Store all the characters in a set
captcha_length = []                             # A list to store the length of each captcha
label_name = dataset["label"]
for word in label_name :
    captcha_length.append(len(word))
    for ch in word :
        characters.add(ch)
characters = sorted(characters)                  
char_to_labels = {char:idx for idx, char in enumerate(characters)}  # Map text to numeric labels 
labels_to_char = {val:key for key, val in char_to_labels.items()}   # Map numeric labels to text
max_length = max(Counter(captcha_length).keys())

print("===============================")
print(char_to_labels)
print(labels_to_char)
print("charcaters number: ", len(characters))
print("Maximum length: ", max(Counter(captcha_length).keys()))
print("===============================")

# Prepare the data ------------------------------------------------------------------------------------
# Split into training and validation sets  
training_data, validation_data = train_test_split(dataset, test_size=0.1, random_state=seed)                           
training_data = training_data.reset_index(drop=True)
validation_data = validation_data.reset_index(drop=True)                                   
print("Number of training samples: ", len(training_data))          
print("Number of validation samples: ", len(validation_data))      

# Build training/validation data
training_data, training_labels = util.generate_arrays(df=training_data, characters=characters, img_height=img_height, img_width=img_width)
print("Number of training images: ", training_data.shape)           #(7650, 72, 100)
print("Number of training labels: ", training_labels.shape)         #(7250,)
validation_data, validation_labels = util.generate_arrays(df=validation_data, characters=characters, img_height=img_height, img_width=img_width)
print("Number of validation images: ", validation_data.shape)
print("Number of validation labels: ", validation_labels.shape)

# Get a generator object for the training/validation data
train_data_generator = util.DataGenerator(data=training_data,
                                     labels=training_labels,
                                     char_map=char_to_labels,
                                     characters=characters,
                                     batch_size=batch_size,
                                     img_width=img_width,
                                     img_height=img_height,
                                     downsample_factor=downsample_factor,
                                     max_length=max_length,
                                     shuffle=True
                                    )
valid_data_generator = util.DataGenerator(data=validation_data,
                                     labels=validation_labels,
                                     char_map=char_to_labels,
                                     characters=characters,
                                     batch_size=batch_size,
                                     img_width=img_width,
                                     img_height=img_height,
                                     downsample_factor=downsample_factor,
                                     max_length=max_length,
                                     shuffle=False
                                    )

# Build the model ------------------------------------------------------------------------------------
model = util.build_model(characters=characters, img_width=img_width, img_height=img_height, max_length=max_length)
plot_model(model, show_shapes=True, show_layer_names=True,to_file='model.png')
model.summary()
save_dir=f"./Saved_Model/"
model_filepath = "model1.h5"
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)  # Add early stopping
checkpoint = ModelCheckpoint(os.path.join(save_dir, model_filepath), monitor='val_loss', verbose=0, save_best_only=True, mode='min')

# Train the model ------------------------------------------------------------------------------------
history = model.fit(train_data_generator,
                    validation_data=valid_data_generator,
                    epochs=train_epoch,
                    callbacks=[es, checkpoint])
model = load_model(f"{save_dir}/{model_filepath}", custom_objects={'CTCLayer' : util.CTCLayer})
prediction_model = keras.models.Model(model.get_layer(name='input_data').input, model.get_layer(name='dense2').output)
prediction_model.summary()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('CTC loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.savefig('history.png')

# Check results on val ------------------------------------------------------------------------------------
labels_to_char[-1]=""
for p, (inp_value, _) in enumerate(valid_data_generator):    
    bs = inp_value['input_data'].shape[0]                       # inp_value = batch_inputs(dict)
    X_data = inp_value['input_data']
    labels = inp_value['input_label']
    preds = prediction_model.predict(X_data)        
    pred_texts = util.predictions_decoding(preds, characters, labels_to_char=labels_to_char)    
    orig_texts = []
    for label in labels:            
        text = ''.join([labels_to_char[int(x)] for x in label])                
        orig_texts.append(text)        
    for i in range(bs):
        print(f'Ground truth: {orig_texts[i]} \t Predicted: {pred_texts[i]}')
    break



