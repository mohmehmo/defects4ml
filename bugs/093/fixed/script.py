from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import numpy as np


inputBatch = np.empty([24,30])
ponlabel = np.eye(24, 1)

model=Sequential()
inputBatch = inputBatch.reshape(24,30, 1)
model.add(Conv1D(64, 3, activation='relu', input_shape=(30, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=4,strides=None, padding='valid'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(inputBatch,ponlabel,batch_size=24,epochs=20,validation_data=(inputBatch, ponlabel))
