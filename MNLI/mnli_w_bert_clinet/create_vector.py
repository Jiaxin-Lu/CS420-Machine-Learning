from bert_serving.client import BertClient
from load_data import test_df, train_df, train_labels, train_texts, test_labels, test_texts
import numpy as np
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization, Dense
import pandas as pd
import os
import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
""" create vector """
bc = BertClient()
x_test = np.array(bc.encode(test_texts))
x_train = np.array(bc.encode(train_texts))

np.savetxt("x_test_vec_fine_5.txt", x_test, delimiter=',')
print(x_test)

np.savetxt("x_train_vec_fine_5.txt", x_train, delimiter=',')
print(x_train)

y_train = np.array([vec for vec in train_df['label']])
np.savetxt("y_train.txt", y_train, delimiter=',')
print("y_train complete")


""" balance labels """
# count = np.zeros(3)
#
# def balance_train():
#     x_train_ = []
#     y_train_ = []
#     for i in range(9000):
#         count[train_labels[i]] += 1
#         if train_labels[i] == 0 and count[0] <= 2300:
#             x_train_.append(x_train[i])
#             y_train_.append(y_train[i])
#         elif train_labels[i] == 1 and count[1] <= 2700:
#             x_train_.append(x_train[i])
#             y_train_.append(y_train[i])
#         elif train_labels[i] == 2 and count[2] <= 3000:
#             x_train_.append(x_train[i])
#             y_train_.append(y_train[i])
#
#     return np.array(x_train_), np.array(y_train_)
#
#
# x_train, y_train = balance_train()
#
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
#
# np.savetxt("x_train_balanced.txt", x_train, delimiter=',')
# np.savetxt("y_train_balanced.txt", y_train, delimiter=',')


