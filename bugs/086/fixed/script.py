from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, BatchNormalization, LeakyReLU, Flatten
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
import numpy as np

X_train = np.empty([6000,10,4])
y_train = np.eye(6000, 1)

X_test = np.empty([1,10,4])
y_test = np.array([1])

model = Sequential()
model.add(Conv1D(input_shape = (10, 4),
                        filters=16,
                        kernel_size=4,
                        padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(Conv1D(filters=8,
                        kernel_size=4,
                        padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())

model.add(Dense(1))
model.add(Activation('sigmoid'))

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.9, patience=30, min_lr=0.000001, verbose=0)

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, 
          nb_epoch = 100, 
          batch_size = 128, 
          verbose=0, 
          validation_data=(X_test, y_test),
          callbacks=[reduce_lr],
          shuffle=True)
# ///////// Added By @Mehdi
import numpy as np
import json
loss = history.history['loss'][-1]
file = open(file="fixed/result.json", mode="w")  
model_loss = np.float64(loss)
res = {"loss" : loss}
json.dump(res, file)
file.close()
# //////////////////////////
y_pred = model.predict(X_test)
