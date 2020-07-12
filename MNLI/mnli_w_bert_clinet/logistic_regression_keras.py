from mnli_w_bert_clinet.load_vector import x_train, y_train, x_test, y_test
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.optimizers import SGD, Adam
import os
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


TOTAL_TEST_BATCH_SIZE = 8000
BATCH_SIZE = 400
N_CLASSES = 3
VECTOR_SIZE = 768
LR = 0.008
# ITERATIONS = 90000
NUM_BATCH = TOTAL_TEST_BATCH_SIZE // BATCH_SIZE
EPOCH = 150


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


def next_train_batch(num):
    left = num * BATCH_SIZE % TOTAL_TEST_BATCH_SIZE
    right = left + BATCH_SIZE
    return x_train[left:right], y_train[left:right]


def next_dev_batch(num):
    left = num * 1000 % TOTAL_TEST_BATCH_SIZE
    right = left + 1000
    return x_train[left:right], y_train[left:right]


# def MLP(dropout_rate=0.25, activation='relu'):
#     start_neurons = 512
#     model = Sequential()
#     model.add(Dense(start_neurons, input_dim=768, activation=activation))
#     model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#
#     model.add(Dense(start_neurons//2, activation=activation))
#     model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#
#     model.add(Dense(start_neurons//4, activation=activation))
#     model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate))
#
#     model.add(Dense(start_neurons//8, activation=activation))
#     model.add(BatchNormalization())
#     model.add(Dropout(dropout_rate/2))
#
#     model.add(Dense(N_CLASSES, activation='softmax'))
#     return model
#
#
# model = MLP(dropout_rate=0.5, activation='relu')
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#
# print("TRAINING ---")
#
# model_train = model.fit(x_train, y_train, validation_data=[x_dev, y_dev],
#                         epochs=EPOCH, batch_size=BATCH_SIZE, shuffle=True, verbose=1)
#
# print("start predict")


model = Sequential()
# model.add(Dense(512, input_shape=(VECTOR_SIZE,), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(N_CLASSES, activation='softmax'))

# model.add(Flatten())
model.add(Dense(input_shape=(VECTOR_SIZE,), units=N_CLASSES))
model.add(Activation('softmax'))
# model.add(Dense(units=243, activation='tanh', input_shape=(VECTOR_SIZE,)))
# model.add(Dense(units=81, activation='tanh'))
# model.add(Dense(units=3, activation='softmax'))

# model.add(Dense(768, activation='relu', init='normal', input_shape=(VECTOR_SIZE,), W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(384, activation='relu', init='normal', W_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Dense(N_CLASSES, activation='softmax'))


model.summary()

model.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

model_train = model.fit(x_train, y_train, epochs=EPOCH, validation_data=[x_dev, y_dev],
                        batch_size=BATCH_SIZE, shuffle=True, verbose=2)


model.save('logistic_regression_keras_model_2.h5')

loss_, accuracy_ = model.evaluate(x_test, y_test)
print("final: loss : %f, accuracy : %f" % (loss_, accuracy_))


# print("start predict")

# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Dense(4, input_shape=(VECTOR_SIZE,), activation='relu')
#         tf.keras.layers.Dense(4, activation='relu')
#         tf.keras.layers.Dense(1, activation='softmax')
#     ]
# )

# y_pred = model.predict_classes(x_test)

# print(y_pred)


# np.savetxt("predicts_keras.txt", y_pred, delimiter=',')
#
# map_back_o = {3: "unknown", 0: "neutral", 1: "entailment", 2: "contradiction"}
# map_back = {3: "unknown", 0: "contradiction", 1: "neutral", 2: "entailment"}
# predicts = []
# for i in range(1000):
#     predicts.append(map_back[y_pred[i]])
# name = ['label']
# out = pd.DataFrame(columns=name, data=predicts)
# out.to_csv("predict_keras.csv", encoding='utf-8')



