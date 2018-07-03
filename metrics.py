import tensorflow as tf


def prediction_status(y_true, y_pred, weighted=True):
  axis = 0 if weighted else None
  tp = tf.count_nonzero(y_pred * y_true, axis=axis)
  tn = tf.count_nonzero((y_pred - 1) * (y_true - 1), axis=axis)
  fn = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)
  fp = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)

  if not weighted:
    return tp, tn, fn, fp, None
  else:
    count_class = tf.reduce_sum(y_true, axis=0)
    weights = count_class / tf.reduce_sum(count_class)

    return tp, tn, fn, fp, weights


def accuracy(y_true, y_pred):
  correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, axis=1), tf.float32),
                                tf.cast(tf.argmax(y_true, axis=1), tf.float32))
  return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def precision_weighted(y_true, y_pred):
  tp, tn, fn, fp, w = prediction_status(y_true, y_pred)

  precision = tp / (tp + fp)
  return tf.reduce_sum(precision * w)


def recall_weighted(y_true, y_pred):
  tp, tn, fn, fp, w = prediction_status(y_true, y_pred)

  recall = tp / (tp + fn)
  return tf.reduce_sum(recall * w)


def f1_weighted(y_true, y_pred):
  prec, reca = precision_weighted(y_true, y_pred), recall_weighted(y_true, y_pred)
  return 2 * prec * reca / (prec + reca)
