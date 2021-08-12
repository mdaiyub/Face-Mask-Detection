# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 23:02:24 2021

@author: Aiyub
"""
#Import Library & Packages
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import cv2

#Image Preprocessing
train_data = 'Face Mask Dataset/Train'
validation_data = 'Face Mask Dataset/Validation'

IMG_SIZE = [224, 224]
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

#Arguments
train_datagen = ImageDataGenerator(rescale = 1./225,
                                  samplewise_center = True,
                                  samplewise_std_normalization = True,
                                  rotation_range = 10,
                                  horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1./225)
train_generator = train_datagen.flow_from_directory(directory = train_data,
                                                        target_size = tuple(IMG_SIZE),
                                                        batch_size = 32,
                                                        shuffle = True,
                                                        class_mode = 'binary')
val_generator = val_datagen.flow_from_directory(directory = validation_data,
                                                        target_size = tuple(IMG_SIZE),
                                                        batch_size = 32,
                                                        shuffle = False,
                                                        class_mode = 'binary')
#Data Visualization
masked_images = os.listdir('Face Mask Dataset/Train/WithMask')
unmasked_images = os.listdir('Face Mask Dataset/Train/WithoutMask')

plt.figure(figsize = (9, 2))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(plt.imread(os.path.join(train_data + "/WithMask",masked_images[i])))
    plt.title("With Mask")

plt.figure(figsize = (9, 2))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(plt.imread(os.path.join(train_data + "/WithoutMask",unmasked_images[i])))
    plt.title("Without Mask")
    
#MobileNetV2 Architechure
mobilenet = MobileNetV2(input_shape = (224, 224, 3), include_top = False)
for layer in mobilenet.layers:
    layer.trainable = False
    
mobilenet.summary()

#Adding Base Layers to the Model
X = AveragePooling2D(pool_size=(7, 7))(mobilenet.output)
X = Flatten(name="flatten")(X)
X = Dense(64, activation="relu")(X)
X = Dropout(0.5)(X)
prediction = Dense(1, activation= 'sigmoid')(X)
model = Model(inputs = mobilenet.input, outputs = prediction)

#Model Compilation
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

#Model Training
history = model.fit(train_generator,
                    epochs = 20,
                    validation_data= val_generator,
                    verbose = 1)

#Save for future
model.save('COVID-19 Face_Mask_Detector.h5')

#Prediction on Model
pred_img = 'Face Mask Dataset/Validation/WithoutMask/1176.png'
image = tf.keras.preprocessing.image.load_img(path=pred_img,
                                             target_size=tuple(IMG_SIZE))

input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  
predictions = model.predict(input_arr)
print(np.around(predictions))

if np.around(predictions) == 1:
    plt.title('Without Mask')
else:
    plt.title("With Mask")
plt.imshow(plt.imread(pred_img))

















