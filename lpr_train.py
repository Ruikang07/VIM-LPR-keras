from lpr_model import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy import load, save
import math, os
from sklearn.model_selection import train_test_split

# load images and labels
images = load('data/AOLP_noSkew_plate_images.npy', allow_pickle=True)
labels = load('data/AOLP_noSkew_plate_labels_in_number.npy', allow_pickle=True)


len_images = len(images)
len_labels = len(labels)

images = images / 255.0


# split data:
tr_images, val_images, tr_labels, val_labels = train_test_split(
    images,
    labels,
    train_size=1056,
    #test_size=0.3,
)
 
 
len_tr = len(tr_images)
len_val = len(val_images) 


if len_tr % BatchSize != 0:
    len_tr = BatchSize * math.floor(len_tr / BatchSize)
    tr_images = tr_images[0:len_tr]
    tr_labels = tr_labels[0:len_tr]
    
   
if len_val % BatchSize != 0:
    len_val = BatchSize *  math.floor(len_val / BatchSize)
    val_images = val_images[0:len_val]
    val_labels = val_labels[0:len_val]    


#separate the label for each char in a plate

tr_label_0 = tr_labels[:,0]
tr_label_1 = tr_labels[:,1]
tr_label_2 = tr_labels[:,2]
tr_label_3 = tr_labels[:,3]
tr_label_4 = tr_labels[:,4]
tr_label_5 = tr_labels[:,5]

val_label_0 = val_labels[:,0]
val_label_1 = val_labels[:,1]
val_label_2 = val_labels[:,2]
val_label_3 = val_labels[:,3]
val_label_4 = val_labels[:,4]
val_label_5 = val_labels[:,5]


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

  
# compile the model
model.compile(
    optimizer=Adam(),
    loss={
        'char0': 'sparse_categorical_crossentropy',
        'char1': 'sparse_categorical_crossentropy',
        'char2': 'sparse_categorical_crossentropy',
        'char3': 'sparse_categorical_crossentropy',
        'char4': 'sparse_categorical_crossentropy',
        'char5': 'sparse_categorical_crossentropy'
    },
    # set the loss weights
    loss_weights={
        'char0': 2.0,
        'char1': 1.0,
        'char2': 1.0,
        'char3': 1.0,
        'char4': 1.0,
        'char5': 2.0
    },
    # select the metrics to evaluate the model
    metrics={
        'char0': ['sparse_categorical_accuracy'],
        'char1': ['sparse_categorical_accuracy'],
        'char2': ['sparse_categorical_accuracy'],
        'char3': ['sparse_categorical_accuracy'],
        'char4': ['sparse_categorical_accuracy'],
        'char5': ['sparse_categorical_accuracy']
    },
)


if not os.path.isdir("./weights") : 
    os.mkdir('./weights')
    
checkpoint_save = tf.keras.callbacks.ModelCheckpoint( 
    filepath = './weights/lpr_best_model_weights_epoch{epoch:03d}.h5',
    save_best_only=True, 
    save_weights_only=True,
    monitor='val_loss', 
    mode='auto'
)



# fit the model
history = model.fit(
    # training data 
    x={
        'image': tr_images
    },
    # training target
    y={
        'char0': tr_label_0,
        'char1': tr_label_1,
        'char2': tr_label_2,
        'char3': tr_label_3,
        'char4': tr_label_4,
        'char5': tr_label_5
    },
    epochs=200, 
    batch_size=BatchSize, 
    validation_data=(
        {
            'image': val_images
        }, 
        {
            'char0': val_label_0,
            'char1': val_label_1,
            'char2': val_label_2,
            'char3': val_label_3,
            'char4': val_label_4,
            'char5': val_label_5
        }
    ), 
    callbacks=[
        checkpoint_save
    ],
    verbose=1,
)


# Saving weights of model
model.save_weights('./weights/lpr_last_model_weights.h5')
