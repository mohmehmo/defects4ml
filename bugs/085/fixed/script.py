from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

rows,cols=x_train.shape[1:]
in_shape=(rows,cols,1)

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

number_of_classes = 1
batch_size = 20
epochs = 5

cnn = Sequential()

cnn.add(Conv2D(64, (5, 50),
padding="same",
activation="relu",data_format="channels_last",
input_shape=in_shape))

cnn.add(MaxPooling2D(pool_size=(2,2),data_format="channels_last"))

cnn.add(Flatten())
cnn.add(Dropout(0.5))

cnn.add(Dense(number_of_classes, activation="softmax"))

cnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

cnn.fit(x_train, y_train,
     batch_size=batch_size,
     epochs=epochs,
     validation_data=(x_test, y_test)
     ,shuffle=True)
