from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxPooling2D, Flatten, Conv2D
import numpy as np
from keras.utils.np_utils import to_categorical   

X_3 = np.empty([5000, 1, 20, 56])

c_y = np.random.randint(0,9,(5000,10))

X_test = np.empty([10, 1, 20, 56])
y_test = np.random.randint(0,9,(10,10))

num_classes = 10

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu',input_shape=(1, 20, 56)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_3, y_3, batch_size=100, epochs=20, verbose=2, validation_data=(X_test, y_test))
