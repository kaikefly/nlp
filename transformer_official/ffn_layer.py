"""Implementation of fully connected network."""

import tensorflow as tf


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_size, filter_size, relu_dropout):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout

    def build(self, input_shape):
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.filter_size,
            use_bias=True,
            activation=tf.nn.relu,
            name='filter_layer')
        self.output_dense_layer = tf.keras.layers.Dense(
            self.hidden_size, use_bias=True, name='output_layer')

    def get_config(self):
        return {
            'hidden_size': self.hidden_size,
            'filter_size': self.filter_size,
            'relu_dropout': self.relu_dropout
        }

    def call(self, x, training):
        """Return outputs of the feedforward network."""
        # x [batch_size, length, hidden_size]
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.output_dense_layer(output)

        return output


def test_ffn_layer():
    hidden_size = 64
    filter_size = 32
    relu_dropout = 0.5
    layer = FeedForwardNetwork(hidden_size, filter_size, relu_dropout)
    print(layer.get_config())
    length = 2
    x = tf.ones([1, length, hidden_size])
    y = layer(x, training=True)
    print(y.shape)


if __name__ == '__main__':
    test_ffn_layer()
