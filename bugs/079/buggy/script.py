import tensorflow as tf

tfe = tf.contrib.eager
tf.enable_eager_execution()

import numpy as np


def make_model():
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(4, activation='relu'))
    net.add(tf.keras.layers.Dense(1))
    return net

def compute_loss(pred, actual):
    return tf.reduce_mean(tf.square(tf.subtract(pred, actual)))

def compute_gradient(model, pred, actual):
    """compute gradients with given noise and input"""
    with tf.GradientTape() as tape:
        loss = compute_loss(pred, actual)
    grads = tape.gradient(loss, model.variables)
    return grads, loss

def apply_gradients(optimizer, grads, model_vars):
    optimizer.apply_gradients(zip(grads, model_vars))

model = make_model()
optimizer = tf.train.AdamOptimizer(1e-4)

x = np.linspace(0,1,1000)
y = x+np.random.normal(0,0.3,1000)
y = y.astype('float32')
train_dataset = tf.data.Dataset.from_tensor_slices((y.reshape(-1,1)))

epochs = 2# 10
batch_size = 25
itr = y.shape[0] // batch_size
for epoch in range(epochs):
    for data in tf.contrib.eager.Iterator(train_dataset.batch(25)):
        preds = model(data)
        grads, loss = compute_gradient(model, preds, data)
        print(grads)
        apply_gradients(optimizer, grads, model.variables)
#         with tf.GradientTape() as tape:
#             loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(preds, data))))
#         grads = tape.gradient(loss, model.variables)
#         print(grads)
#         optimizer.apply_gradients(zip(grads, model.variables),global_step=None)

