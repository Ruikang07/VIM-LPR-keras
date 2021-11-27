from lpr_model import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy import load
import cv2 

model = Model(
    #inputs={'image': in_image},
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

PateBase = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

#image = cv2.imread('sample_plate_img/AOLP_AC_51.jpg')
image = cv2.imread('sample_plate_img/AOLP_LE_654.jpg')

img2 = cv2.resize(image, (160,64))
img_arr = np.array(img2)

# reshape data for the model
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
for c1 in y:
    str1.append(PateBase[c1])    
print("\nplate_pred on_train_batch_begin = ", str1)
