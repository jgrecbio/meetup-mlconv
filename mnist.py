from math import floor
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

data = tf.keras.datasets.mnist.load_data(".mnist.npz")

(feature_train, label_train), (feature_test, label_test) = data

label_enc = LabelBinarizer()
y_train = label_enc.fit_transform(label_train).astype(np.float32)
x_train = feature_train.reshape((60000, 28 * 28)).astype(np.float32)
x_test = feature_test.reshape((-1, 28 * 28)).astype(np.float32)
y_test = label_test.astype(np.float32)


print(feature_train.shape)
print(label_train.shape)
print(x_train.shape)
print(y_train.shape)

x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])
y_true_test = tf.placeholder(tf.float32, [None])

w = tf.Variable(tf.random_normal(shape=(784, 10)))
b = tf.Variable(tf.random_normal(shape=(10,)))
y_pred = tf.matmul(x, w) + b

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, axis=1), tf.float32), y_true_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

nb_epoch = 1000
batch_size = 512
n_batch = int(floor(x_train.shape[0] / batch_size))
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(1000):
    # sess.run(train_step, feed_dict={x: x_train, y_true: y_train})
    for b in range(n_batch):
      sess.run(train_step, feed_dict={x: x_train[b * batch_size: (b + 1) * batch_size],
                                      y_true: y_train[b * batch_size: (b + 1) * batch_size]})

    acc = sess.run(accuracy, feed_dict={x: x_test,
                                        y_true_test: y_test})
    print(i, acc)
