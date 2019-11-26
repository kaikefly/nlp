import tensorflow as tf
import attention_layer
import embedding_layer
import ffn_layer
import metrics
import model_utils

EOS_ID = 1


def create_model(params, is_train):
    """Create transformer model"""
    with tf.name_scope('model'):
        if is_train:
            inputs = tf.keras.layers.Input((None,), dtype='int64', name='inputs')
            targets = tf.keras.layers.Input((None,), dtype='int64', name='targets')
            internal_model = Transformer(params, name='transformer_v2')
            logits = internal_model([inputs, targets], training=is_train)
            vocab_size = params['vocab_size']
            label_smoothing = params['label_smoothing']
            if params['enable_metrics_in_training']:
                logits = metrics.MetricLayer(vocab_size)([logits, targets])
            logits = tf.keras.layers.Lambda(lambda x: x, name='logits',
                                            dtype=tf.float32)(logits)
            model = tf.keras.Model([inputs, targets], logits)
            loss = metrics.transformer_loss(
                logits, targets, label_smoothing, vocab_size)
            model.add_loss(loss)
            return model
        else:
            inputs = tf.keras.layers.Input((None,), dtype='int64', name='inputs')
            internal_model = Transformer(params, name='transformer_v2')
            ret = internal_model([inputs], training=is_train)
            outputs, scores = ret['outputs'], ret['scores']
            return tf.keras.Model(inputs, [outputs, scores])


class Transformer(tf.keras.Model):
    """Transformer model with keras"""
    def __init__(self, params, name=None):
        super(Transformer, self).__init__(name=name)
        self.params = params
        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            params['vocab_size'], params['hidden_size'])
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, inputs, training):
        if len(inputs) == 2:
            inputs, targets = inputs[0], inputs[1]
        else:
            inputs, targets = inputs[0], None

        with tf.name_scope('Transformer'):
            attention_bias = model_utils.get_padding_bias(inputs)
            encoder_outputs = self.encode(inputs, attention_bias, training)
            if targets is None:
                return self.predict(encoder_outputs, attention_bias, training)
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias, training)
                return logits

    def encode(self, inputs, attention_bias, training):
        """Generate continuous representation for inputs.
        inputs: [batch_size, input_length]
        attention_bias: [batch_size, 1, 1, input_length]
        return: [batch_size, input_length, hidden_size]
        """
        with tf.name_scope('encode'):
            embedded_inputs = self.embedding_softmax_layer(inputs)
            embedded_inputs = tf.cast(embedded_inputs, self.params['dtype'])
            inputs_padding = model_utils.get_padding(inputs)
            attention_bias = tf.cast(attention_bias, self.params['dtype'])

            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params['hidden_size'])
                pos_encoding = tf.cast(pos_encoding, self.params['dtype'])
                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.params['layer_postprocess_dropout'])

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding, training=training)

    def decode(self, targets, encoder_outputs, attention_bias, training):
        """Generate logits for each value in the target sequence
        targets: [batch_size, target_length]
        encoder_outputs: [batch_size, input_length, hidden_size]
        attention_bias: [batch_size, 1, 1, input_length]
        return: [batch_size, target_length, vocab_size]
        """
        with tf.name_scope('decode'):
            decoder_inputs = self.embedding_softmax_layer(targets)
            decoder_inputs = tf.cast(decoder_inputs, self.params['dtype'])
            attention_bias = tf.cast(attention_bias, self.params['dtype'])
            with tf.name_scope('shift_targets'):
                decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            with tf.name_scope('add_pos_encoding'):
                length = tf.shape(decoder_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params['hidden_size'])
                pos_encoding = tf.cast(pos_encoding, self.params['dtype'])
                decoder_inputs += pos_encoding
            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.params['layer_postprocess_dropout'])

            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length, dtype=self.params['dtype'])
            outputs = self.decoder_stack(decoder_inputs,
                                         encoder_outputs,
                                         decoder_self_attention_bias,
                                         attention_bias,
                                         training=training)
            logits = self.embedding_softmax_layer(outputs, mode='linear')
            logits = tf.cast(logits, tf.float32)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""




class LayerNormalization(tf.keras.layers.Layer):
    """Applies layer normalization"""
    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__(dtype='float32')
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.scale = self.add_weight(
            'layer_norm_scale',
            shape=[self.hidden_size],
            initializer=tf.ones_initializer())
        self.bias = self.add_weight(
            'layer_norm_bias',
            shape=[self.hidden_size],
            initializer=tf.zeros_initializer())
        super(LayerNormalization, self).build(input_shape)

    def get_config(self):
        return {
            'hidden_size': self.hidden_size
        }

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, params):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = params['layer_postprocess_dropout']

    def build(self, input_shape):
        self.layer_norm = LayerNormalization(self.params['hidden_size'])
        super(PrePostProcessingWrapper, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, x, *args, **kwargs):
        training = kwargs['training']
        y = self.layer_norm(x)
        y = self.layer(y, *args, **kwargs)

        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y


class EncoderStack(tf.keras.layers.Layer):
    """Transformer encoder stack.
    The encoder stack is made up of N identical layers. Each layer is composed of the sublayers:
    1. self-attention layer
    2. feedforward network (which is 2 fully-connected layers)
    """
    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        params = self.params
        for _ in range(params['num_hidden_layers']):
            self_attention_layer = attention_layer.SelfAttention(
                params['hidden_size'], params['num_heads'],
                params['attention_dropout'])
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                params['hidden_size'], params['filter_size'], params['relu_dropout'])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params),
                PrePostProcessingWrapper(feed_forward_network, params)
            ])

        # create final layer normalization layer.
        self.output_normalization = LayerNormalization(params['hidden_size'])
        super(EncoderStack, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, encoder_inputs, attention_bias, inputs_padding, training):
        """Return the output of the encoder layer stacks.
        encoder_inputs: [batch_size, input_length, hidden_size]
        attention_bias: [batch_size, 1, 1, input_length]
        inputs_padding: [batch_size, input_length]
        Return: [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.name_scope('layer_%d' % n):
                with tf.name_scope('self_attention'):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias, training=training)
                with tf.name_scope('ffn'):
                    encoder_inputs = feed_forward_network(encoder_inputs, training=training)

        return self.output_normalization(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
    """Transformer decoder stack.
    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer if composed of the sublayers:
    1. self-attention layer
    2. multi-headed attention layer combining encoder outputs with results from the previous self-attention layer
    3. feedward network (2 fully-connected layers)
    """
    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        params = self.params
        for _ in range(params['num_hidden_layers']):
            self_attention_layer = attention_layer.SelfAttention(
                params['hidden_size'], params['num_heads'], params['attention_dropout'])
            enc_dec_attention_layer = attention_layer.Attention(
                params['hidden_size'], params['num_heads'], params['attention_dropout'])
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                params['hidden_size'], params['filter_size'], params['relu_dropout'])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params),
                PrePostProcessingWrapper(enc_dec_attention_layer, params),
                PrePostProcessingWrapper(feed_forward_network, params)
            ])
        self.output_normalization = LayerNormalization(params['hidden_size'])
        super(DecoderStack, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, training):
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            layer_name = 'layer_%d' % n
            with tf.name_scope(layer_name):
                with tf.name_scope('self_attention'):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs, decoder_self_attention_bias,training=training)
                with tf.name_scope('encdec_attention'):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs,
                        encoder_outputs,
                        attention_bias,
                        training=training)
                with tf.name_scope('ffn'):
                    decoder_inputs = feed_forward_network(
                        decoder_inputs, training=training)

        return self.output_normalization(decoder_inputs)
