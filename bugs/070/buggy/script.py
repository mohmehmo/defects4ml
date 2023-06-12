from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import optimizers


# fix random seed for reproducibility
seed = 7
#datapoints
X = np.arange(0.0, 5.0, 0.1, dtype='float32').reshape(-1,1)
y = 5 * np.power(X,2) + np.power(np.random.randn(50).reshape(-1,1),3)

#model
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Activation('relu'))
model.add(Dense(30, init='uniform'))
model.add(Activation('relu'))
model.add(Dense(output_dim=1))
model.add(Activation('linear'))


#training
sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
history = model.fit(X, y, nb_epoch=1000)

# ///////// Added By @Mehdi
import numpy as np
import json
loss = history.history['loss'][-1]
file = open(file="buggy/result.json", mode="w")  
model_loss = np.float64(loss)
res = {"loss" : model_loss}
json.dump(res, file)
file.close()


#predictions
predictions = model.predict(X)

#plot
# plt.scatter(X, y,edgecolors='g')
# plt.plot(X, predictions,'r')
# plt.legend([ 'Predictated Y' ,'Actual Y'])
# plt.show()
# sys.exit(1)
