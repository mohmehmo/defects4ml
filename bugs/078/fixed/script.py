import numpy as np
import random

# generate random data with two features
n_samples = 200
n_feats = 2

cls0 = np.random.uniform(low=0.2, high=0.4, size=(n_samples,n_feats))
cls1 = np.random.uniform(low=0.5, high=0.7, size=(n_samples,n_feats))
x_train = np.concatenate((cls0, cls1))
y_train = np.concatenate((np.zeros((n_samples,)), np.ones((n_samples,))))

# shuffle data because all negatives (i.e. class "0") are first
# and then all positives (i.e. class "1")
indices = np.arange(x_train.shape[0])
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


model = Sequential()
model.add(Dense(2, activation='sigmoid', input_shape=(n_feats,)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-1),
              metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=True)

import numpy as np
import json
accuracy = history.history['accuracy'][-1]
file = open(file="fixed/result.json", mode="w")  
model_accuracy = np.float64(accuracy)
res = {"accuracy" : model_accuracy}
json.dump(res, file)
file.close()