import numpy as np
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization, Dense
import pandas as pd
import os
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


""" load vector """
x_train = np.loadtxt("x_train_vec_fine_5.txt", delimiter=',')
x_test = np.loadtxt("x_test_vec_fine_5.txt", delimiter=',')
# x_train = np.loadtxt("x_train_vec_base.txt", delimiter=',')
# x_test = np.loadtxt("x_test_vec_base.txt", delimiter=',')
# x_train = np.loadtxt("x_train_vec_large.txt", delimiter=',')
# x_test = np.loadtxt("x_test_vec_large.txt", delimiter=',')
y_train = np.loadtxt("y_train.txt", delimiter=',')
y_test = np.loadtxt("y_test.txt", delimiter=',')



