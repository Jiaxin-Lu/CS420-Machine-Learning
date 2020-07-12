from mnli_w_bert_clinet.load_vector import x_train, y_train, x_test, y_test
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, Conv1D, \
    MaxPooling1D
from keras.optimizers import SGD
import os
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


TOTAL_TEST_BATCH_SIZE = 8000
BATCH_SIZE = 400
N_CLASSES = 3
VECTOR_SIZE = 768
LR = 0.02
# ITERATIONS = 90000
NUM_BATCH = TOTAL_TEST_BATCH_SIZE // BATCH_SIZE
EPOCH = 8


y_train = to_categorical(y_train, num_classes=N_CLASSES, dtype='float32')
y_test = to_categorical(y_test, num_classes=N_CLASSES, dtype='float32')

print(y_train.shape)


def do_permutation(x_, y_):
    x_t = []
    y_t = []
    permutation = list(np.random.permutation(9000))
    for i in permutation:
        x_t.append(x_[i])
        y_t.append(y_[i])
    return np.array(x_t), np.array(y_t)


# x_train, y_train = do_permutation(x_train, y_train)

x_dev = x_train[8000:9000]
y_dev = y_train[8000:9000]
x_train = x_train[0:8000]
y_train = y_train[0:8000]

x_dev = x_dev.reshape(-1, VECTOR_SIZE, 1)
x_train = x_train.reshape(-1, VECTOR_SIZE, 1)
x_test = x_test.reshape(-1, VECTOR_SIZE, 1)


def next_train_batch(num):
    left = num * BATCH_SIZE % TOTAL_TEST_BATCH_SIZE
    right = left + BATCH_SIZE
    return x_train[left:right], y_train[left:right]


def next_dev_batch(num):
    left = num * 1000 % TOTAL_TEST_BATCH_SIZE
    right = left + 1000
    return x_train[left:right], y_train[left:right]


model = Sequential()
model.add(Conv1D(256, 5, padding='same', activation='relu', input_shape=(VECTOR_SIZE, 1)))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(128, 5, padding='same'))
model.add(MaxPooling1D(3, 3, padding='same'))
model.add(Conv1D(64, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

model_train = model.fit(x_train, y_train, epochs=EPOCH, validation_data=[x_dev, y_dev],
                        batch_size=BATCH_SIZE, shuffle=True, verbose=2)


model.save('CNN-keras-model.h5')

loss_, accuracy_ = model.evaluate(x_test, y_test)
print("final: loss : %f, accuracy : %f" % (loss_, accuracy_))



