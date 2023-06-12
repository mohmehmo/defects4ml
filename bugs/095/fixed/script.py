import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import seaborn as sns
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split

train_data_path = "sonar.all-data" #path where data is stored
train_data = pd.read_csv(train_data_path, header=None) #load data in dataframe using pandas


X_train, X_val = train_test_split(train_data, test_size=0.2, random_state=42)

X_train, y_train = X_train.iloc[:, :-1], X_train.iloc[:, -1]
X_val, y_val = X_val.iloc[:, :-1], X_val.iloc[:, -1]
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

print(f"type of y is {type(y_train)}")
y_train = y_train.map({'R': 1, 'M': 0})
y_val = y_val.map({'R': 1, 'M': 0})

def make_model():
    input_vec = Input((60,))
    final = Dense(12, activation='relu')(input_vec)
    final = Dense(1, activation='sigmoid')(final)

    model = Model(inputs=[input_vec], outputs=[final])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = make_model()
model.summary()

history = model.fit(X_train, y_train, validation_data=[X_val, y_val], epochs=10, batch_size=64)

# ///////// Added By @Mehdi
import numpy as np
import json
accuracy = history.history['accuracy'][-1]
file = open(file="result.json", mode="w")  
model_accuracy = np.float64(accuracy)
res = {"accuracy" : model_accuracy}
json.dump(res, file)
file.close()
# //////////////////////////

# print(f"Accuracy of model is {history.history['accuracy'][-1]}")

# print(model.evaluate(X_train, y_train))
# print(model.evaluate(X_val, y_val))
