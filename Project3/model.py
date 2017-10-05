"""
Udacity Self Driving Car Nanodegree Project 3.

Behavorial Cloning.

"""

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import MaxPooling2D, BatchNormalization, Cropping2D, Conv2D

# Read in data from disk
data_file = 'D:\CarND_Data\\UD\driving_log.csv'
lines = []
with open(data_file) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)


# Load data for processing
images = []
measurements = []

source_path = 'D:\CarND_Data\\UD\IMG\\'

for line in lines:
    if line[4] != 0:  # Don't include when speed = 0
        filename = line[0].split('/')[1]
        image = cv2.imread(source_path + filename)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)


# Add flipped images to the dataset
augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# Build Model - This model is based on NVidia model https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

input_shape=(160, 320, 3)
dropout_prob = 0.2
batch_size = 128
epochs = 8
activation = 'relu'
validation_split = 0.2

model = Sequential()
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=input_shape))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(24, (5, 5),  strides=(2, 2),activation=activation))
model.add(Dropout(dropout_prob))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation=activation))
model.add(Dropout(dropout_prob))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation=activation))
model.add(Dropout(dropout_prob))
model.add(Conv2D(64, (3, 3), activation=activation))
model.add(Dropout(dropout_prob))
model.add(Conv2D(64, (3, 3), activation=activation))
model.add(MaxPooling2D())
model.add(Dropout(dropout_prob))
model.add(Flatten())
model.add(Dense(1162, activation=activation))
model.add(Dense(100, activation=activation))
model.add(Dense(50, activation=activation))
model.add(Dense(10, activation=activation))
model.add(Dense(1))


# Train model
print('Start model training.')
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=True)
print('Model training complete.')

model.save('model.h5')
print('Model saved')