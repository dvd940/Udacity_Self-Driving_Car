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
import matplotlib.pyplot as plt

# Read in data from disk
data_file = 'D:\CarND_Data\\UD\driving_log.csv'
lines = []
with open(data_file) as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)


# Load data for processing
print('Processing images')
images = []
measurements = []
steering_offset = [0, 0.25, -0.25]  # steering offset for center, left and right images

source_path = 'D:\CarND_Data\\UD\IMG\\'

for line in lines:
    if float(line[6]) >= 1.0:  # Don't include when speed less than 1
        for i in range(0,3):        
            filename = line[i].split('/')[1]        
            image = cv2.imread(source_path + filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB color space to match drive.py         
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement + steering_offset[i])

        # add image flipped 180 and flipped sterring angle if not zero steering angle
        if measurement != 0:
            images.append(cv2.flip(image, 1))  
            measurements.append(measurement*-1.0)  

X_train = np.array(images)
y_train = np.array(measurements)


# Build Model - This model is based on NVidia model https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

input_shape=(160, 320, 3)
dropout_prob = 0.2
batch_size = 128
epochs = 7
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
history_object = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, shuffle=True)
print('Model training complete.')
model.save('model.h5')
print('Model saved')


# Plot training and validation loss and save to disk
plt.rcParams["figure.figsize"] = (25,10)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])

plt.title('Model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right', fontsize=20)
plt.savefig('loss.png')