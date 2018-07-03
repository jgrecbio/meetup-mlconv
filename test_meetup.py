import unittest
import numpy as np
import tensorflow as tf
from metrics import accuracy, precision_weighted, recall_weighted, \
  f1_weighted
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelBinarizer


class TestMeetup(unittest.TestCase):
  def test_metrics(self):
    y_true = np.random.randint(size=(100,), low=0, high=10)
    y_pred = np.random.randint(size=(100, ), low=0, high=10)

    enc = LabelBinarizer()
    enc.fit(np.concatenate((y_true, y_pred)))
    y_true, y_pred = enc.fit_transform(y_true), enc.fit_transform(y_pred)

    sk_acc = accuracy_score(y_true, y_pred)
    sk_prec = precision_score(y_true, y_pred, average="weighted")
    sk_reca = recall_score(y_true, y_pred, average="weighted")
    sk_f1 = f1_score(y_true, y_pred, average="weighted")

    tf_true, tf_pred = tf.constant(y_true), tf.constant(y_pred)
    tf_acc = accuracy(tf_true, tf_pred)
    tf_prec = precision_weighted(y_true, y_pred)
    tf_reca = recall_weighted(y_true, y_pred)
    tf_f1 = f1_weighted(y_true, y_pred)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      acc, prec, reca, f1 = sess.run([tf_acc, tf_prec, tf_reca, tf_f1])

    # print(acc)
    # self.assertAlmostEqual(sk_acc, acc)
    self.assertTrue(np.allclose(sk_prec, prec))
    self.assertTrue(np.allclose(sk_reca, reca))
    self.assertTrue(np.allclose(sk_f1, f1, rtol=0.01))
