import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=5, data_format="channels_last", activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(data_format="channels_last"))
model.add(Conv2D(filters=16, kernel_size=3, data_format="channels_last", activation="relu"))
model.add(MaxPooling2D(data_format="channels_last"))

model.add(Flatten(data_format="channels_last"))

model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=5, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

x = np.load("./data/x.npy")
y = np.load("./data/y.npy")


history = model.fit(x=x, y=y, batch_size=100, epochs=20, validation_split=0.2)


# ///////// Added By @Mehdi
import json
import numpy as np
vall_acc = history.history["val_accuracy"][-1]

file = open(file="buggy/result.json", mode="w")  
model_acc = np.float64(vall_acc)
res = {"val_acc" : model_acc}
json.dump(res, file)
file.close()
