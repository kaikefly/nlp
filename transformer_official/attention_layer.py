"""Implementation of multihead attention and self-attention layers."""

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""
    def __init__(self, hidden_size, num_heads, attention_dropout):
        if hidden_size % num_heads:
            raise ValueError(
                'Hidden size ({}) must be divisible by the number of heads ({}).'
                .format(hidden_size, num_heads))

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        """Layers for linearly projecting the queries, keys, and values"""
        self.q_dense_layer = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name='q')
        self.k_dense_layer = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name='k')
        self.v_dense_layer = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name='v')
        self.output_dense_layer = tf.keras.layers.Dense(self.hidden_size, use_bias=False, name='output_transform')
        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'attention_dropout': self.attention_dropout
        }

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
        x: [batch_size, length, hidden_size]
        return: [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with tf.name_scope('split_heads'):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            # calculate depth of last dimension after it has been split.
            depth = self.hidden_size // self.num_heads
            # split the last dimension
            x = tf.reshape(x, [batch_size, length, self.num_heads, depth])
            # transpose the result
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
        x: [batch_size, num_heads, length, hidden_size/num_heads]
        return: [batch_size, length, hidden_size]
        """
        with tf.name_scope('combine_heads'):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[2]
            x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
            return tf.reshape(x, [batch_size, length, self.hidden_size])

    def call(self, x, y, bias, training, cache=None):
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            key = tf.concat([tf.cast(cache['k'], k.dtype), k], axis=1)
            value = tf.concat([tf.cast(cache['v'], v.dtype), v], axis=1)
            cache['k'] = key
            cache['v'] = value

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # scale q to prevent the dot product between q and k from growing too large.
        depth = self.hidden_size // self.num_heads
        q *= depth ** -0.5

        logits = tf.matmul(q, k, transpose_b=True)
        logits += bias
        weights = tf.nn.softmax(logits, name='attention_weights')
        if training:
            weights = tf.nn.dropout(weights, 1.0 - self.attention_dropout)
        attention_output = tf.matmul(weights, v)

        # recombine heads-> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""
    def call(self, x, bias, training, cache=None):
        return super(SelfAttention, self).call(x, x, bias, training, cache)


def test_attention_layer():
    hidden_size = 64
    num_heads = 4
    dropout = 0.5
    depth = hidden_size // num_heads

    layer = SelfAttention(hidden_size, num_heads, dropout)
    print(layer.get_config())
    length = 2
    x = tf.ones([1, length, hidden_size])
    bias = tf.ones([1])
    y = layer(x, bias, training=True)
    print(y.shape)


if __name__ == '__main__':
    test_attention_layer()
