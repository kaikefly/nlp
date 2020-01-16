"""Transformer model demo."""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
from transformer import Transformer

MAX_LENGTH = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64


examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)


params = {
    'input_vocab_size': tokenizer_pt.vocab_size + 2,
    'target_vocab_size': tokenizer_en.vocab_size + 2,
    'hidden_size': 512,
    'num_hidden_layers': 6,
    'num_heads': 8,
    'attention_dropout': 0.1,
    'filter_size': 2048,
    'relu_dropout': 0.1,
    'alpha': 0.6,
    'dtype': tf.float32,
    'layer_postprocess_dropout': 0.1,
    'extra_decode_length': 10,
    'beam_size': 3
}


def encode(lang1, lang2):
    """
    Add a start and end token to the input and target.
    """
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


def tf_encode(pt, en):
    return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# Cache the dataset to memory to get a speedup while reading from it
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(params['hidden_size'])
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

transformer = Transformer(params, name='transformer_v2')

checkpoint_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(model=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Lastest checkpoint restored!')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64)
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        logits = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, logits)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, logits)


def train():
    for epoch in range(20):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                            train_loss.result(), train_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def evaluate(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]

    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(40):
        logits = transformer([encoder_input, output], training=False)
        predictions = logits[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == tokenizer_en.vocab_size + 1:
            return tf.squeeze(output, axis=0)

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def translate(sentence):
    outputs = evaluate(sentence)
    predicted_sentence = tokenizer_en.decode([i for i in outputs if i < tokenizer_en.vocab_size])
    print(predicted_sentence)


if __name__ == '__main__':
    # train()
    translate('este é um problema que temos que resolver.')
    print("Real translation: this is a problem we have to solve .")
    translate("os meus vizinhos ouviram sobre esta ideia.")
    print("Real translation: and my neighboring homes heard about this idea .")
    translate("vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.")
    print("Real translation: so i 'll just share with you some stories very quickly of some magical things that have happened .")
    translate("este é o primeiro livro que eu fiz.")
    print("Real translation: this is the first book i've ever done.")
