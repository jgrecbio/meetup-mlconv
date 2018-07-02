import time
from math import floor
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

data = tf.keras.datasets.mnist.load_data(".mnist.npz")
nb_epoch = 10
batch_size = 512

(feature_train, label_train), (feature_test, label_test) = data

label_enc = LabelBinarizer()
label_train = label_enc.fit_transform(label_train).astype(np.float32)
feature_train = feature_train.reshape((60000, 28 * 28)).astype(np.float32)
feature_test = feature_test.reshape((-1, 28 * 28)).astype(np.float32)
label_test = label_test.astype(np.float32)

n_batch = int(floor(feature_train.shape[0] / batch_size))

train = tf.data.Dataset
train = train.from_tensor_slices((feature_train, label_train))
train = train.repeat()
train = train.batch(batch_size)
train = train.make_one_shot_iterator()

test = tf.data.Dataset
test = test.from_tensor_slices((feature_test, label_test))
test = test.repeat()
test = test.batch(feature_test.shape[0])
test = test.make_one_shot_iterator()

x_train, y_train = train.get_next()
x_test, y_test = test.get_next()

w = tf.Variable(tf.random_normal(shape=(784, 10)))
b = tf.Variable(tf.random_normal(shape=(10,)))
y_pred_train = tf.matmul(x_train, w) + b
y_pred_test = tf.matmul(x_test, w) + b

cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=y_pred_train)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred_test, axis=1), tf.float32), y_test)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

start = time.time()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(10):
    for b in range(n_batch):
      a = sess.run(x_test).shape
      sess.run(train_step)

    acc = sess.run(accuracy)
    print(i, acc)

end = time.time()
time_length = end - start
print(time_length)
