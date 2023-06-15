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
print("Tensorflow version: ", tf.__version__)
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
random.seed(seed)


if (tf.test.is_gpu_available()):
    fraction = 0.8
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    # config.gpu_options.per_process_gpu_memory_fraction = fraction
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
else :
    os.environ["CUDA_VISIBLE_DEVICES"]='-1' 


batch_size = 16         # Batch size for training and validation
img_width = 100         # Desired image dimensions
img_height = 72         # Desired image dimensions
downsample_factor = 4   # Factor by which the image is going to be downsampled by the convolutional blocks
max_length = 4          # Maximum length of any captcha in the data


# Read testing data ------------------------------------------------------------------------------------
dataset_path_ = "./dataset"
data_dir = Path(f"{dataset_path_}/test/")
images = list(data_dir.glob("task*/*.png"))         # Get list of all the images
print("Number of images found: ", len(images))

dataset_path = []
dataset_name = []
for i in range(len(images)):
    dataset_path.append(str(images[i]))
    name = str(images[i]).split("test/")[1]
    dataset_name.append(name)

dataset_dummy_label = []
for i in range(len(dataset_path)):
    dataset_dummy_label.append('0')     # Store image-label info
content = {
    "filepath" : dataset_path,
    "filename" : dataset_name ,
    "label" : dataset_dummy_label
    }
dataset = pd.DataFrame(content)
print(dataset.head())
# dataset = dataset.sample(frac=1.).reset_index(drop=True) # Shuffle the dataset
testing_data = dataset.reset_index(drop=True)
print(testing_data.head())


dataset_temp = pd.read_csv(f"{dataset_path_}/train/annotations.csv")  # Training data -> for char_to_labels
characters = set()                      # Store all the characters in a set
captcha_length = []                     # A list to store the length of each captcha
label_name = dataset_temp["label"]
for word in label_name :
    captcha_length.append(len(word))
    for ch in word :
        characters.add(ch)
characters = sorted(characters)         # Sort the characters   
char_to_labels = {char:idx for idx, char in enumerate(characters)}  # Map text to numeric labels 
labels_to_char = {val:key for key, val in char_to_labels.items()}   # Map numeric labels to text

print("===============================")
print(char_to_labels)
print(labels_to_char)
print("charcaters number: ", len(characters))
print("Maximum length: ", max(Counter(captcha_length).keys()))
print("===============================")

# Prepare the data ------------------------------------------------------------------------------------
# Build testing data
testing_data, testing_labels = util.generate_arrays(df=testing_data, characters=characters, img_height=img_height, img_width=img_width)
print("Number of testing images: ", testing_data.shape)
print("Number of testing labels: ", testing_labels.shape)
     
# Get a generator object for the training data
testing_data_generator = util.DataGenerator(data=testing_data,
                                     labels=testing_labels,
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
class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred, input_length, label_length):
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return loss

save_dir=f"./Saved_Model/"
filepath = "model1.h5"
model = load_model(f"{save_dir}/{filepath}", custom_objects={'CTCLayer' : CTCLayer})
prediction_model = keras.models.Model(model.get_layer(name='input_data').input, model.get_layer(name='dense2').output)
prediction_model.summary()

# Check results on test ------------------------------------------------------------------------------------
labels_to_char[-1]=""
count = 0
orig_texts = []
pred_texts = []

for p, (inp_value, _) in enumerate(testing_data_generator):    # <util.DataGenerator object at 0x7f7e6859b5c0>    
    bs = inp_value['input_data'].shape[0]
    X_data = inp_value['input_data']
    preds = prediction_model.predict(X_data)        
    pred_texts_temp = util.predictions_decoding(preds, characters=characters, labels_to_char=labels_to_char)
    for txt in pred_texts_temp :
        pred_texts.append(txt)
    
# print(pred_texts_temp)
# print(pred_texts)
    
    # for label in labels:            
        # text = ''.join([labels_to_char[int(x)] for x in label])                
        # orig_texts.append(text)        
                
# for i in range(len(orig_texts)):    
    # if(orig_texts[i]==pred_texts[i]) : 
        # count+=1
    # else :
        # print(f'Ground truth: {orig_texts[i]} \t Predicted: {pred_texts[i]}')    
# print(f"Accuracy : {count/testing_data.shape[0]}")

df = pd.DataFrame(list(zip(dataset_name, pred_texts)), columns =['filename', 'label']) 
df.to_csv('Answer16.csv', index=False)
    