from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, MaxPooling2D, Flatten, Conv2D
import numpy as np
from keras.utils.np_utils import to_categorical   

X_3 = np.empty([5000, 20, 56, 1])

y_3 = np.random.randint(0,9,(5000,10))

X_test = np.empty([10, 20, 56, 1])
y_test = np.random.randint(0,9,(10,10))


print(X_3.shape)
print(y_3.shape)
num_classes = 10

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu',input_shape=(20, 56,1)))
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
