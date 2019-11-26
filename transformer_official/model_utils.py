import tensorflow as tf
import math

_NEG_INF_FP32 = -1e9


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding
    return: [length, hidden_size]
    """
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) / (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_decoder_self_attention_bias(length, dtype=tf.float32):
    """防止 decoder 偷看答案的 mask
    return: [1, 1, length, length]
    """
    neg_inf = _NEG_INF_FP32
    with tf.name_scope('decoder_self_attention_bias'):
        valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype), -1, 0)  # 下三角矩阵
        valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
        decoder_bias = neg_inf * (1.0 - valid_locs)
    return decoder_bias


def get_padding(x, padding_value=0, dtype=tf.float32):
    with tf.name_scope('padding'):
        return tf.cast(tf.equal(x, padding_value), dtype)


def get_padding_bias(x, padding_value=0, dtype=tf.float32):
    """Calculate bias tensor from padding values in tensor.
    x: [batch_size, length]
    return: [batch_size, 1, 1, length]
    """
    with tf.name_scope('attention_bias'):
        padding = get_padding(x, padding_value, dtype)
        attention_bias = padding * _NEG_INF_FP32
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1)
    return attention_bias
