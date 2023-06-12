# Imports
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, Reshape
from keras import Model
# from matplotlib import pyplot as plt
import numpy as np

# Constants definition
numClasses = 10
inputShape = (28, 28)

# Dataset import
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Model definition
inputs = Input(shape=inputShape)
x = Reshape((28, 28, 1))(inputs)
x = Conv2D(4, 3, activation='relu', padding='same')(x)
x = MaxPool2D()(x)
x = Conv2D(4, 3, activation='relu', padding='same')(x)
x = MaxPool2D()(x)
x = Conv2D(8, 3, activation='relu', padding='same')(x)
x = MaxPool2D()(x)
x = Conv2D(8, 3, activation='relu', padding='same')(x)
x = MaxPool2D()(x)
x = Flatten()(x)
y = Dense(32, activation='relu')(x)
out = Dense(numClasses, activation='softmax')(y)
model = Model(inputs=inputs, outputs=out)

# Print model detail
model.summary()

# Configure model optimizer and metrics
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, batch_size=200)

# Evaluate model
loss = model.evaluate(x_test, y_test, batch_size=32)
print('Loss: %.4f\tAccuracy: %.4f' % tuple(loss))

eval_acc = loss[1]

# ///////// Added By @Mehdi
import json
import numpy as np
file = open(file="fixed/result.json", mode="w")  
model_acc = np.float64(eval_acc)
res = {"eval_acc" : model_acc}
json.dump(res, file)
file.close()
# //////////////////////

# # Showing prediction result
# test_data = np.reshape(x_test[0], (1, 28, 28))
# prediction = model.predict(test_data)
# prediction = np.argmax(prediction)

# print('Prediction result:', prediction)
# test_data = np.reshape(x_test[0], (28, 28))
# plt.imshow(test_data)
# plt.show()
