from keras.datasets import mnist
from keras.layers import Conv2D, Dense
from keras.models import Sequential
from tensorflow.keras import activations

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
model = Sequential([
    Conv2D(8, kernel_size=(3, 3), padding="same", activation=activations.relu, input_shape=(28, 28, 1)),
    Dense(64, activation=activations.relu),
    Dense(64, activation=activations.relu),
    Dense(10, activation=activations.softmax)
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train, y_train, epochs=5) #error