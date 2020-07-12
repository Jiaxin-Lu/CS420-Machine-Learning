from mnli_w_bert_clinet.load_vector import x_train, y_train, x_test, y_test
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
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
EPOCH = 150

y_test_ = y_test
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


y_pred = []
y_accuracy = []

for _ in range(3):
    x_train_, y_train_ = do_permutation(x_train, y_train)

    x_dev = x_train_[8000:9000]
    y_dev = y_train_[8000:9000]
    x_train_ = x_train_[0:8000]
    y_train_ = y_train_[0:8000]

    model = Sequential()
    model.add(Dense(input_shape=(VECTOR_SIZE,), units=N_CLASSES))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(optimizer=SGD(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    model_train = model.fit(x_train_, y_train_, epochs=EPOCH, validation_data=[x_dev, y_dev],
                            batch_size=BATCH_SIZE, shuffle=True, verbose=2)

    # model.save('logistic_regression_keras_model_2.h5')

    loss_, accuracy_ = model.evaluate(x_test, y_test)
    y_accuracy.append(accuracy_)
    print("final: loss : %f, accuracy : %f" % (loss_, accuracy_))
    y_pred.append(model.predict_classes(x_test))


y_prediction = []
for i in range(1000):
    ls = np.zeros(3)
    for _ in range(3):
        ls[int(y_pred[_][i])] += y_accuracy[_]
    ans_ = np.argmax(ls)
    y_prediction.append(ans_)


tot = 0
for i in range(1000):
    if y_prediction[i] == y_test_[i]:
        tot += 1
print("final accuracy %f" % (tot / 1000))

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



