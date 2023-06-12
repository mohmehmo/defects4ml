from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy
import time
import keras
import sys
#fix random seed for reproducibility
numpy.random.seed(7)
batch_size =10
#load and read dataset
dataset = numpy.genfromtxt("Book1.csv", delimiter=',')
# split into input (X) and output (Y) variables
X = dataset[:,2:4]
Y = dataset[:,1]
print ("Variables: \n", X)
print ("Target_outputs: \n", Y)
# create model
model = Sequential()
model.add(Dense(4, input_dim=2))
model.add(Activation('relu'))
#model.add(Dense(4, activation='relu'))
model.add(Dense(1))
model.add(Activation('tanh'))
model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['MSE'])
# Fit the model
model.fit(X, Y, epochs=50, batch_size=1)
F = model.predict(X)

 # ///////// Added By @Mehdi
import numpy as np
import json
file = open(file="result.json", mode="w")  
model_prediction = np.float64(F)
res = {"prediction" : model_prediction.tolist()}
json.dump(res , file)
file.close()
# ///////////////////
