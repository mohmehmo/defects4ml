from keras.layers import Input, Lambda, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from keras import backend as K

x = Input(shape=(30,3))
low = np.random.rand(30, 3)
high = 1 + np.random.rand(30, 3)
clipped_out_position = Lambda(lambda x, low, high: K.tf.clip_by_value(x, K.constant(low), K.constant(high)),
                              arguments={'low': low, 'high': high})(x)

model = Model(inputs=x, outputs=[clipped_out_position])
optimizer = Adam(lr=.1)
model.compile(optimizer=optimizer, loss="mean_squared_error")
checkpoint = ModelCheckpoint("debug.hdf", monitor="val_loss", verbose=1, save_best_only=True, mode="min")
training_callbacks = [checkpoint]
model.fit(np.random.rand(100, 30, 3), [np.random.rand(100, 30, 3)], callbacks=training_callbacks, epochs=50, batch_size=10, validation_split=0.33)
