import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('mathtext', default='regular')


learning_rate = [0.1, 0.07, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.001]
loss = [1.082833, 0.819573, 0.802196, 0.792377, 0.789415, 0.783790, 0.776774, 0.775693, 0.803458]
accuracy = [0.646, 0.658, 0.656, 0.656, 0.666, 0.664, 0.668, 0.643, 0.629]

learning_rate = np.array(learning_rate)
loss = np.array(loss)
accuracy = np.array(accuracy)

x = range(len(learning_rate))
# y1 = np.arrange(np.min(accuracy), np.max(accuracy))
# y2 = np.arrange(np.min(loss), np.max(loss))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, accuracy, '-', label='Accuracy')
ax1.set_ylabel('Accuracy')
# ax1.legend(loc=0)
ax1.set_xlabel('Learning rates')

ax2 = ax1.twinx()
ax2.plot(x, loss, '-g', label='Loss')
ax2.set_ylabel('Loss')
# ax2.legend(loc=0)

plt.xticks(x, learning_rate)
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

plt.savefig("learning_rate_test_keras_lr.png")
plt.show()

# plt.plot(x, loss, label='Loss')
# plt.plot(x, accuracy, label='Accuracy')
# plt.xticks(x, learning_rate)


