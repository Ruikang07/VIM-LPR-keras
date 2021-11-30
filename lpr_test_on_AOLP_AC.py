from lpr_model import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy import load, save
import math, os
import cv2 

# load images and labels
images = load('data/AOLP_AC_plate_images.npy', allow_pickle=True)
labels = load('data/AOLP_AC_plate_labels_in_number.npy', allow_pickle=True)


len_images = len(images)
len_labels = len(labels)

print("len_images = ", len_images)
print("len_labels = ", len_labels)

#construct the lpr model
model = Model(
    inputs={'image': inputs},
    outputs={
        'char0': y0,
        'char1': y1,
        'char2': y2,
        'char3': y3,
        'char4': y4,
        'char5': y5
    },
)

model.load_weights("pretrained_model/lpr_model_weights.h5")

num_char_correct = 0
num_char_total = 0

for i in range(len_images): 
    image = images[i]
    img2 = cv2.resize(image, (160,64))
    img_arr = np.array(img2)

    image = img_arr.reshape((1, 64, 160, 3))

    image = image / 255.0

    string_pred1 = model(image )

    y = []
    y.append(int(tf.math.argmax(string_pred1['char0'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char1'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char2'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char3'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char4'], 1)))
    y.append(int(tf.math.argmax(string_pred1['char5'], 1)))

    str1 = []
    for j in range(6):
        num_char_total += 1
        if y[j] == labels[i][j]:
            num_char_correct += 1

accuracy = 100.* float( num_char_correct) / num_char_total
print("The accuracy of the pretrained model on AOLP AC dataset is = ", accuracy,"%")