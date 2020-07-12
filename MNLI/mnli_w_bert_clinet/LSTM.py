import numpy as np
import tensorflow as tf
import pandas as pd
from mnli_w_bert_clinet.load_vector import x_train, y_train, x_test, y_test
from keras.utils import to_categorical
import matplotlib.pyplot as plt


BATCH_SIZE = 100
TIME_STEP = 24
INPUT_SIZE = 32
LR = 0.005
NUM_UNITS = 100
ITERATIONS = 3000
N_CLASSES = 3

TOTAL_TEST_BATCH_SIZE = 8000

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
    left = num * 2 % TOTAL_TEST_BATCH_SIZE
    right = left + 1000
    return x_train[left:right], y_train[left:right]


train_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])
image = tf.reshape(train_x, [-1, TIME_STEP, INPUT_SIZE])
train_y = tf.placeholder(tf.int32, [None, N_CLASSES])

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS)
outputs, final_state = tf.nn.dynamic_rnn(
    cell=rnn_cell,
    inputs=image,
    initial_state=None,
    dtype=tf.float32,
    time_major=False
)

loss_history = []

output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_CLASSES)
loss = tf.losses.softmax_cross_entropy(onehot_labels=train_y, logits=output)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

correct_prediction = tf.equal(tf.argmax(train_y, axis=1), tf.argmax(output, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("start")
for step in range(ITERATIONS):
    x, y = next_train_batch(step)
    _, loss_ = sess.run([train_op, loss], {train_x: x, train_y: y})
    loss_history.append(loss_)
    if step % 500 == 0:
        # x_, y_ = x_dev, y_dev
        accuracy_ = sess.run(accuracy, {train_x: x_dev, train_y: y_dev})
        print("step : %d , loss : %f , accuracy : %f" % (step, loss_, accuracy_))
        accuracy_ = sess.run(accuracy, {train_x: x_test, train_y: y_test})
        print("step : %d , test accuracy : %f" % (step, accuracy_))

accuracy_ = sess.run(accuracy, {train_x: x_train, train_y: y_train})
print("final, train accuracy : %f" % accuracy_)
accuracy_ = sess.run(accuracy, {train_x: x_dev, train_y: y_dev})
print("final, dev accuracy : %f" % accuracy_)
accuracy_ = sess.run(accuracy, {train_x: x_test, train_y: y_test})
print("final, test accuracy : %f" % accuracy_)


def print_loss(loss_h):
    plt_x = range(len(loss_h))
    plt.plot(plt_x, loss_h, label='Loss')
    plt.legend()
    plt.xlabel('Iteration Num')
    plt.ylabel('Loss')
    plt.savefig("bert_vec_LSTM.png")
    plt.show()


print_loss(loss_history)


def predict_result():
    pred_prob = sess.run(output, {train_x: x_test})
    pred_ans = tf.argmax(pred_prob, axis=1)
    pred_ans_numpy = pred_ans.eval(session=sess)

    print(pred_ans_numpy)
    np.savetxt("predicts.txt", pred_ans_numpy, delimiter=',')

    map_back = {3: "unknown", 0: "neutral", 1: "entailment", 2: "contradiction"}
    predicts = []
    for i in range(1000):
        predicts.append(map_back[pred_ans_numpy[i]])
    name = ['label']
    out = pd.DataFrame(columns=name, data=predicts)
    out.to_csv("predict.csv", encoding='utf-8')


# predict_result()


# model = Sequential()


# ans = np.argmax(model.predict(x_test), axis=1)
# print(ans)
# np.savetxt("predicts.txt", ans, delimiter=',')
#
# map_back = {3: "unknown", 0: "neutral", 1: "entailment", 2: "contradiction"}
# predicts = []
# for i in range(1000):
#     predicts.append(map_back[ans[i]])
# name = ['label']
# out = pd.DataFrame(columns=name, data=predicts)
# out.to_csv("predict.csv", encoding='utf-8')
