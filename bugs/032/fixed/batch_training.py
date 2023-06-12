import sys
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import Sequence
from deeplab import DeepLabV3Plus
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

smooth = K.epsilon()
# image_dir = 'images'
# mask_dir = 'masks'

image_dir = 'data/images'
mask_dir = 'data/mask_uncropped'

image_list = os.listdir(image_dir)
mask_list = os.listdir(mask_dir)
image_list.sort()
mask_list.sort()
print(
    f'*** Number of images: {len(image_list)} ***\n*** Number of masks: {len(mask_list)} ***')

# sanity check
for i in range(len(image_list)):
    assert image_list[i][:-3] == mask_list[i][:-3]


batch_size = int(sys.argv[1])
samples = len(image_list)
img_height, img_width = 512, 512
classes = 66


# def dice_coef_multilabel(y_true, y_pred, n=66):
#     def dice_coef(y_true, y_pred):
#         y_true_f = K.flatten(y_true)
#         y_pred_f = K.flatten(y_pred)
#         intersection = K.sum(y_true_f * y_pred_f)
#         return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     dice = 0
#     for index in range(n):
#         dice -= dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
#     return dice


def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    J = (intersection + 1.0) / (K.sum(y_true_f) +
                                K.sum(y_pred_f) - intersection + 1.0)
    return 1 - J


def ce_dice_loss(y_true, y_pred, n=66):
    for index in range(n):
        J = 0
        J += jaccard_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return categorical_crossentropy(y_true, y_pred) + J

# def focal_loss1(target, output, gamma=2):
#     output /= K.sum(output, axis=-1, keepdims=True)
#     eps = K.epsilon()
#     output = K.clip(output, eps, 1. - eps)
#     return -K.sum(K.pow(1. - output, gamma) * target * K.log(output), axis=-1)


# def focal_loss(y_true, y_pred):
#     gamma = 2.
#     alpha = 4.
#     epsilon = 1.e-9
#     y_true = tf.convert_to_tensor(y_true, tf.float32)
#     y_pred = tf.convert_to_tensor(y_pred, tf.float32)
#     model_out = tf.add(y_pred, epsilon)
#     ce = tf.multiply(y_true, -tf.log(model_out))
#     weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
#     fl = tf.multiply(alpha, tf.multiply(weight, ce))
#     reduced_fl = tf.reduce_max(fl, axis=1)
#     return tf.reduce_mean(reduced_fl)


h, w = img_height, img_width
with open('data/config.json', 'r') as f:
    config = json.load(f)
labels = config['labels']
color_map = {}
for label in labels:
    color_map[label['readable']] = label['color']
label_list = sorted(color_map.keys())
id_to_color = {v: color_map[k] for v, k in enumerate(label_list)}
id_to_label = {v: k for v, k in enumerate(label_list)}


def pipeline(image):
    alpha = 0.5
    dims = image.shape
    x = cv2.resize(image, (w, h))
    x = preprocess_input(np.float32(x))
    z = model.predict(np.expand_dims(x, axis=0))
    z = np.squeeze(z)
    z = cv2.resize(z, (dims[1], dims[0]))
    y = np.argmax(z, axis=2)

    img_color = image.copy()
    for i in np.unique(y):
        img_color[y == i] = id_to_color[i]
    cv2.addWeighted(image, alpha, img_color, 1 - alpha, 0, img_color)
    return img_color


def make_image(tensor):
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(np.uint8(tensor))
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardImage(Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        print(f'viz running')
        val_list = image_list[19500:]
        i = [13, 355, 456, 132, 172, 200]
        for idx in i:
            print(f'*** saving viz {idx} ***')
            test = load_img(f'{image_dir}/{val_list[idx]}')
            test = img_to_array(test)
            segmap = pipeline(test)
            cv2.imwrite(f'train_outputs/{idx}_{epoch}.png',
                        cv2.cvtColor(segmap, cv2.COLOR_RGB2BGR))

            segmap = make_image(segmap)
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=self.tag+f'_{i}', image=segmap)])
            writer = tf.summary.FileWriter('./logs')
            writer.add_summary(summary, epoch)
            writer.close()


class seg_gen(Sequence):
    def __init__(self, x_set, y_set, batch_size, name):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        print(f'*** Initializating {name} generator ***')

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    # def __getitem__(self, idx):
    #     idx = np.random.randint(0, samples, batch_size)
    #     batch_x, batch_y = [], []
    #     for i in idx:
    #         _image = image.img_to_array(image.load_img(
    #             f'{image_dir}/{image_list[i]}', target_size=(img_height, img_width)))
    #         _image = preprocess_input(_image)
    #         mask = image.img_to_array(image.load_img(
    #             f'{mask_dir}/{mask_list[i]}', color_mode='grayscale', target_size=(img_height, img_width)))
    #         mask = to_categorical(mask, num_classes=classes)

    #         batch_y.append(mask)
    #         batch_x.append(_image)
    #     return np.array(batch_x), np.array(batch_y)

    def __getitem__(self, idx):
        idx = np.random.randint(0, samples, batch_size)
        batch_x, batch_y = [], []
        for i in idx:
            _image = image.img_to_array(image.load_img(
                f'{image_dir}/{image_list[i]}'))
            _image = preprocess_input(_image)
            mask = image.img_to_array(image.load_img(
                f'{mask_dir}/{mask_list[i]}', color_mode='grayscale'))

            hh = _image.shape[0]
            ww = _image.shape[1]

            maxh = np.random.randint(img_height, hh)
            maxw = maxh

            rnh = np.random.randint(0, hh - maxh)
            rnw = np.random.randint(0, hh - maxw)

            _image = _image[rnh:rnh + maxh, rnw:rnw + maxw, :]
            mask = mask[rnh:rnh + maxh, rnw:rnw + maxw, :]

            _image = cv2.resize(_image, (img_width, img_height))
            mask = cv2.resize(mask, (img_width, img_height))
            # mask = mask.reshape(img_height, img_width, 1)
            mask = to_categorical(mask, num_classes=66)
            batch_y.append(mask)
            batch_x.append(_image)
            del mask
            del _image
        return np.array(batch_x), np.array(batch_y)


def lr_sched(epoch):
    initial_lrate = 1e-4
    drop = 0.5
    epochs_drop = 3.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(f'Setting lr =>{lrate}')
    return lrate


model = DeepLabV3Plus(img_height, img_width, classes)
# try:
#     print('Loading previous weights')
#     model.load_weights('model.h5')
# except:
#     print('Training from scratch !!!')
print('*** Building multi_gpu_model ***')
pmodel = multi_gpu_model(model, 4)
pmodel.compile(optimizer=Adam(lr=1e-4),
               loss=ce_dice_loss, metrics=['acc'])
print('*** Building multi_gpu_model completed ***')


tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max', filepath='top_weights.h5', monitor='val_acc',
                     save_best_only='True', save_weights_only='True', verbose=1)
lrsched = LearningRateScheduler(lr_sched)
viz = TensorBoardImage('validation_image')
callbacks = [mc, viz, tb]

train_gen = seg_gen(image_list[:-500], mask_list[:-500], batch_size, 'train')
val_gen = seg_gen(image_list[-500:], mask_list[-500:], batch_size, 'val')
steps = train_gen.__len__()
val_steps = val_gen.__len__()

print(f'*** Batch size = {batch_size} ***\n*** Steps per epoch = {steps} ***')
print(f'*** Val steps = {val_steps} ***')

# batch_x = []
# batch_y = []
# from tqdm import tqdm
# print('*** Loading images and masks into memory ***')
# for i in tqdm(range(20000)):
#     _image = image.img_to_array(image.load_img(f'{image_dir}/{image_list[i]}', target_size=(img_height, img_width)))
#     _image = preprocess_input(_image)
#     mask = image.img_to_array(image.load_img(f'{mask_dir}/{mask_list[i]}', color_mode='grayscale', target_size=(img_height, img_width)))
#     batch_x.append(_image)
#     batch_y.append(mask)
# batch_y = np.array(batch_y)
# batch_x = np.array(batch_x)


history = pmodel.fit_generator(train_gen, validation_data=val_gen, validation_steps=val_steps, steps_per_epoch=steps, epochs=int(sys.argv[2]),
                     callbacks=callbacks, workers=8, use_multiprocessing=True, max_queue_size=50)
# pmodel.fit(batch_x, batch_y, batch_size=batch_size, epochs=int(sys.argv[2]), validation_split=0.1, callbacks=callbacks)
print('Saving final weights')
model.save_weights('model.h5')

train_loss = history.history["loss"]
loss = train_loss[-1:]
file = open(file="fixed/result.json", mode="w")  
model_loss = np.float64(loss)
res = {"loss" : model_loss}
json.dump(res, file)
file.close()