"""Implementation of embedding layer with shared weights"""

import tensorflow as tf


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size):
        """
        Specify characteristic parameters of embedding layer
        :param vocab_size: Number of tokens in the embedding.
        :param hidden_size: Dimensionality of the embedding.
        """
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        """Build embedding layer"""
        with tf.name_scope('embedding_and_softmax'):
            self.shared_weights = self.add_weight(
                'weights',
                shape=[self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self.hidden_size**-0.5))
        super(EmbeddingSharedWeights, self).build(input_shape)

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size
        }

    def call(self, inputs, mode='embedding'):
        """Get token embeddings of input"""
        # inputs [batch_size, length] for embedding,
        # [batch_size, length, hidden_size] for linear
        if mode == 'embedding':
            return self._embedding(inputs)  # [batch_size, length, embedding_size]
        elif mode == 'linear':
            return self._linear(inputs)  # [batch_size, length, vocab_size]
        else:
            raise ValueError('mode {} is not valid.'.format(mode))

    def _embedding(self, inputs):
        with tf.name_scope('embedding'):
            embeddings = tf.gather(self.shared_weights, inputs)  # 类似于 embedding_lookup [batch_size, length, hidden_size]
            mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
            embeddings *= tf.expand_dims(mask, -1)
            # 论文中提到的权重乘以 sqrt(d_model)
            embeddings *= self.hidden_size ** 0.5

            return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer."""
        with tf.name_scope('presoftmax_linear'):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])  # [batch_size*length, hidden_size]
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)  # [batch_size*length, vocab_size]

            return tf.reshape(logits, [batch_size, length, self.vocab_size])  # [batch_size, length, vocab_size]


def test_embedding_layer():
    vocab_size = 50
    hidden_size = 64
    length = 2
    layer = EmbeddingSharedWeights(vocab_size, hidden_size)
    print(layer.get_config())
    idx = tf.ones([1, length], dtype='int32')
    y = layer(idx)
    print(y.shape)  # [1, length, hidden_size]
    x = tf.ones([1, length, hidden_size])
    output = layer(x, 'linear')
    print(output.shape)  # [1, length, vocab_size]


if __name__ == '__main__':
    test_embedding_layer()
