from math import floor
from toolz import compose
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

x = tf.placeholder(tf.float32, [None, 784], name="x")
y_true = tf.placeholder(tf.float32, [None, 10], "y_true")
y_true_test = tf.placeholder(tf.float32, [None], "y_true_test")


def get_layer(w, b, act):
  def layer(x):
    return act(tf.matmul(x, w) + b)

  return layer


w1 = tf.Variable(tf.random_normal(shape=(784, 100)), name="w1")
w2 = tf.Variable(tf.random_normal(shape=(100, 100)), name="w2")
w3 = tf.Variable(tf.random_normal(shape=(100, 100)), name="w3")
w4 = tf.Variable(tf.random_normal(shape=(100, 10)), name="w4")

b1 = tf.Variable(tf.random_normal(shape=(100,)), name="b1")
b2 = tf.Variable(tf.random_normal(shape=(100,)), name="b2")
b3 = tf.Variable(tf.random_normal(shape=(100,)), name="b3")
b4 = tf.Variable(tf.random_normal(shape=(10,)), name="b4")

for var in [w1, w2, w3, w4, b1, b2, b3, b4]:
  tf.summary.histogram(var.name, var)

layer1 = get_layer(w1, b1, tf.nn.sigmoid)
layer2 = get_layer(w2, b2, tf.nn.sigmoid)
layer3 = get_layer(w3, b3, tf.nn.sigmoid)
layer4 = get_layer(w4, b4, tf.identity)

net = compose(layer4, layer3, layer2, layer1)

y_pred = net(x)

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
consolidated_cost = tf.reduce_mean(cost)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# test
correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, axis=1), tf.float32), y_true_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("testing acuracy", accuracy)
merged = tf.summary.merge_all()

# training summary
cost_sum = tf.summary.scalar("cost", consolidated_cost)
truth_training = tf.cast(tf.argmax(y_true, axis=1), tf.float32)
acc_training = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(y_pred, axis=1),
                                                       tf.float32),
                                               truth_training),
                                      tf.float32))
acc_training_sum = tf.summary.scalar("training accuracy", acc_training)


def training(dir="./.logs"):
  nb_epoch = 1000
  batch_size = 512
  n_batch = int(floor(x_train.shape[0] / batch_size))
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(dir, graph=sess.graph)
    for i in range(nb_epoch):
      for b in range(n_batch):
        _, cost_sum_eval = sess.run([train_step, cost_sum],
                                    feed_dict={x: x_train[b * batch_size: (b + 1) * batch_size],
                                               y_true: y_train[b * batch_size: (b + 1) * batch_size]})

      acc_training_sumed = sess.run(acc_training_sum, feed_dict={x: x_train, y_true: y_train})

      acc, merged_summary = sess.run([accuracy, merged], feed_dict={x: x_test,
                                                                    y_true_test: y_test})
      writer.add_summary(merged_summary, global_step=i)
      writer.add_summary(cost_sum_eval, global_step=i)
      writer.add_summary(acc_training_sumed, global_step=i)
