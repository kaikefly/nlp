"""Functions for calculating loss, accuracy, and other model metrics."""

import functools
import tensorflow as tf


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)"""
    with tf.name_scope('pad_to_same_length'):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])  # [batch_size, length, vocab_size]
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])  # [batch_size, length]
        return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    """Calculate cross entropy loss while ignoring padding
    logits: [batch_size, length_logits, vocab_size]
    labels: [batch_size, length_labels]
    """
    with tf.name_scope('loss'):
        logits, labels = _pad_tensors_to_same_length(logits, labels)

        # calculate smoothing cross entropy
        with tf.name_scope('smoothing_cross_entropy'):
            confidence = 1.0 - smoothing
            low_confidence = smoothing / tf.cast(vocab_size - 1, tf.float32)
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
            xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets)

            normalizing_constant = -(
                confidence * tf.math.log(confidence) +
                tf.cast(vocab_size - 1, tf.float32) * low_confidence *
                tf.math.log(low_confidence + 1e-20))
            xentropy -= normalizing_constant

        weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        return xentropy * weights, weights


def padded_neg_log_perplexity(logits, labels, vocab_size):
    num, den = padded_cross_entropy_loss(logits, labels, 0, vocab_size)
    return -num, den


class MetricLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size):
        super(MetricLayer, self).__init__()
        self.vocab_size = vocab_size
        self.metric_mean_fns = []

    def build(self, input_shape):
        neg_log_perplexity = functools.partial(
            padded_neg_log_perplexity, vocab_size=self.vocab_size)
        self.metric_mean_fns = [
            (tf.keras.metrics.Mean('neg_log_perplexity'), neg_log_perplexity)
        ]
        super(MetricLayer, self).build(input_shape)

    def get_config(self):
        return {'vocab_size': self.vocab_size}

    def call(self, inputs):
        logits, targets = inputs[0], inputs[1]
        for mean, fn in self.metric_mean_fns:
            m = mean(*fn(logits, targets))
            self.add_metric(m)
        return logits


def transformer_loss(logits, labels, smoothing, vocab_size):
    """Calculates total loss containing cross entropy with padding ignored.
    logits: [batch_size, length_logits, vocab_size]
    labels: [batch_size, length_labels]
    """
    xentropy, weights = padded_cross_entropy_loss(logits, labels, smoothing, vocab_size)
    return tf.reduce_sum(xentropy) / tf.reduce_sum(weights)


def test_metric_layer():
    vocab_size = 50
    logits = tf.keras.layers.Input((None, vocab_size),
                                   dtype='float32',
                                   name='logits')
    targets = tf.keras.layers.Input((None,), dtype='int64', name='targets')
    output_logits = MetricLayer(vocab_size)([logits, targets])
    print(output_logits.shape)


if __name__ == '__main__':
    test_metric_layer()
