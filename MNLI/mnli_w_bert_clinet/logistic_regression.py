from mnli_w_bert_clinet.load_vector import x_train, y_train, x_test, y_test
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


TOTAL_TEST_BATCH_SIZE = 8000
BATCH_SIZE = 100
N_CLASSES = 3
# VECTOR_SIZE = 768
VECTOR_SIZE = 1024
LR = 0.008
# ITERATIONS = 90000
NUM_BATCH = TOTAL_TEST_BATCH_SIZE // BATCH_SIZE
EPOCH = 1000


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

print(x_train.shape)


def next_train_batch(num):
    left = num * BATCH_SIZE % TOTAL_TEST_BATCH_SIZE
    right = left + BATCH_SIZE
    return x_train[left:right], y_train[left:right]


def next_dev_batch(num):
    left = num * 1000 % TOTAL_TEST_BATCH_SIZE
    right = left + 1000
    return x_train[left:right], y_train[left:right]


# w = tf.get_variable(name='w', shape=[VECTOR_SIZE, N_CLASSES], initializer=tfl.truncated_normal_initializer(stddev=0.02))
# # b = tf.get_variable(name='b', shape=[N_CLASSES], initializer=tf.zeros_initiaizer())

x = tf.placeholder(tf.float32, [None, VECTOR_SIZE], name='x')
y = tf.placeholder(tf.float32, [None, N_CLASSES], name='y')

w = tf.Variable(tf.random_normal([VECTOR_SIZE, N_CLASSES], stddev=0.02), name='w')
b = tf.Variable(tf.zeros([N_CLASSES]), name='b')
# for train
x_ = tf.nn.dropout(x, keep_prob=0.9)

logits = tf.nn.bias_add(tf.matmul(x_, w), b)
log_logits = tf.nn.log_softmax(logits, axis=-1)
cost = -tf.reduce_mean(tf.reduce_sum(y * log_logits, axis=-1))

# for prediction
logits_prob = tf.nn.bias_add(tf.matmul(x, w), b)
probability = tf.nn.softmax(logits_prob, axis=-1)
pred_ = tf.argmax(probability, axis=1)
acc = tf.equal(pred_, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(acc, tf.float32))


train = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

avg_cost = 0

map_back_o = {3: "unknown", 0: "neutral", 1: "entailment", 2: "contradiction"}
map_back = {3: "unknown", 0: "contradiction", 1: "neutral", 2: "entailment"}

loss_history = []

for i in range(EPOCH):
    avg_cost = 0
    for step in range(NUM_BATCH):
        train_x, train_y = next_train_batch(step)
        sess.run(train, {x: train_x, y: train_y})
        avg_cost += sess.run(cost, {x: train_x, y: train_y}) / NUM_BATCH
    loss_history.append(avg_cost)
    if i % 50 == 0:
        # dev_x, dev_y = next_dev_batch(i//10)
        accuracy_ = sess.run(accuracy, {x: x_dev, y: y_dev})
        print("epoch : %d , accuracy : %f, loss : %f" % (i, accuracy_, avg_cost))
        accuracy_ = sess.run(accuracy, {x: x_test, y: y_test})
        print("epoch : %d , test accuracy : %f" % (i, accuracy_))

    # predict every 1000 iterations
    # if (i+1) % 1000 == 0:
    #     accuracy_ = sess.run(accuracy, {x: x_train, y: y_train})
    #     print("fianal , accuracy : %f" % (accuracy_))
    #
    #     pred_ans_numpy = sess.run(pred_, {x: x_test})
    #     # pred_ans = tf.argmax(pred_prob, axis=1)
    #     # pred_ans_numpy = pred_ans.eval(session=sess)
    #
    #     print(pred_ans_numpy)
    #     file_name = "predicts_%d" % ((i+1) // 1000)
    #     np.savetxt(file_name + ".txt", pred_ans_numpy, delimiter=',')
    #
    #     predicts = []
    #     for t in range(1000):
    #         predicts.append(map_back[pred_ans_numpy[t]])
    #     name = ['label']
    #     out = pd.DataFrame(columns=name, data=predicts)
    #     out.to_csv(file_name + ".csv", encoding='utf-8')


accuracy_ = sess.run(accuracy, {x: x_train, y: y_train})
print("final train : accuracy : %f, loss : %f" % (accuracy_, avg_cost))
accuracy_ = sess.run(accuracy, {x: x_dev, y: y_dev})
print("final dev : accuracy : %f, loss : %f" % (accuracy_, avg_cost))
accuracy_ = sess.run(accuracy, {x: x_test, y: y_test})
print("final test : accuracy : %f, loss : %f" % (accuracy_, avg_cost))


def print_loss(loss_h):
    plt_x = range(len(loss_h))
    plt.plot(plt_x, loss_h, label='Loss')
    plt.legend()
    plt.xlabel('Iteration Num')
    plt.ylabel('Loss')
    plt.savefig('out.png')
    plt.show()


print_loss(loss_history)
