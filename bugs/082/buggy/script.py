import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Activation
from keras.utils import np_utils

#Create model
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(None, 286, 384, 1 )))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

#Create random data
n_images=100
data = np.random.randint(0,2,n_images*286*384)
labels = np.random.randint(0,2,n_images)
labels = np_utils.to_categorical(list(labels))

#add dimension to images
data = data.reshape(n_images,286,384,1)

#Fit model
model.fit(data, labels, verbose=1)	
