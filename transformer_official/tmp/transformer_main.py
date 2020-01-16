"""Train and evaluate the Transformer model."""

import os
import tempfile

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

import data_pipeline
import optimizer
import transformer
import model_params
import translate
from utils import tokenizer

FLAGS = tf.flags.FLAGS

INF = int(1e9)
_SINGLE_SAMPLE = 1


class TransformerTask(object):
    """Main entry of Transformer model."""

    def __init__(self, flags_obj):
        self.flags_obj = flags_obj
        self.predict_model = None

        self.params = params = model_params.BASE_MULTI_GPU_PARAMS.copy()
        params['data_dir'] = flags_obj.data_dir
        params['model_dir'] = flags_obj.model_dir
        params['max_length'] = flags_obj.max_length
        params['decode_batch_size'] = flags_obj.decode_batch_size
        params['decode_max_length'] = flags_obj.decode_max_length
        params['batch_size'] = flags_obj.batch_size or params['decode_batch_size']
        params["num_parallel_calls"] = (
                flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)
        params['repeat_dataset'] = None
        params['dtype'] = tf.float32
        params['num_gpus'] = 0

    def train(self):
        """Trains the model."""
        params = self.params
        flags_obj = self.flags_obj

        _ensure_dir(flags_obj.model_dir)

        model = transformer.create_model(params, is_train=True)
        opt = self._create_optimizer()

        current_step = 0
        checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
        latest_checkpoint = tf.train.latest_checkpoint(flags_obj.model_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logging.info('Loaded checkpoint %s', latest_checkpoint)
            current_step = opt.iterations.numpy()
        model.compile(opt)
        model.summary()

        train_ds = data_pipeline.train_input_fn(params)
        map_data_fn = data_pipeline.map_data_for_transformer_fn
        train_ds = train_ds.map(map_data_fn, num_parallel_calls=params['num_parallel_calls'])

        callbacks = self._create_callbacks(flags_obj.model_dir, 0, params)

        while current_step < flags_obj.train_steps:
            remaining_steps = flags_obj.train_steps - current_step
            train_steps_per_eval = (
                remaining_steps if remaining_steps < flags_obj.steps_between_evals
                else flags_obj.steps_between_evals)
            current_iteration = current_step // flags_obj.steps_between_evals

            logging.info("Start train iteration at global step:{}".format(current_step))
            history = model.fit(
                train_ds,
                initial_epoch=current_iteration,
                epochs=current_iteration + 1,
                steps_per_epoch=train_steps_per_eval,
                callbacks=callbacks,
                # If TimeHistory is enabled, progress bar would be messy. Increase
                # the verbose level to get rid of it.
                verbose=1)
            current_step += train_steps_per_eval
            logging.info("Train history: {}".format(history.history))

            logging.info("End train iteration at global step:{}".format(current_step))

        stats = _build_stats(history, callbacks)
        return stats

    def eval(self):
        """Evaluates the model."""
        if not self.predict_model:
            self.predict_model = transformer.create_model(self.params, False)
        self._load_weights_if_possible(
            self.predict_model,
            tf.train.latest_checkpoint(self.flags_obj.model_dir))
        self.predict_model.summary()

    def predict(self):
        """Predicts result from the model."""
        params = self.params
        flags_obj = self.flags_obj

        with tf.name_scope('model'):
            model = transformer.create_model(params, is_train=False)
            self._load_weights_if_possible(
                model, tf.train.latest_checkpoint(self.flags_obj.model_dir))
            model.summary()
        subtokenizer = tokenizer.Subtokenizer(flags_obj.vocab_file)

        ds = data_pipeline.eval_input_fn(params)
        ds = ds.map(lambda x, y: x).take(_SINGLE_SAMPLE)
        ret = model.predict(ds)
        val_outputs, _ = ret
        length = len(val_outputs)
        for i in range(length):
            translate.translate_from_input(val_outputs[i], subtokenizer)

    def _create_callbacks(self, cur_log_dir, init_steps, params):
        """Creates a list of callbacks."""
        sfunc = optimizer.LearningRateFn(params['learning_rate'],
                                         params['hidden_size'],
                                         params['learning_rate_warmup_steps'])
        scheduler_callback = optimizer.LearningRateScheduler(sfunc, init_steps)
        callbacks = [scheduler_callback]
        ckpt_full_path = os.path.join(cur_log_dir, 'cp-{epoch:04d}.ckpt')
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_full_path, save_weights_only=True))
        return callbacks

    def _load_weights_if_possible(self, model, init_weight_path=None):
        """Loads model weights when it is provided."""
        if init_weight_path:
            logging.info('Load weights: {}'.format(init_weight_path))
            model.load_weights(init_weight_path)
        else:
            logging.info('Weights not loaded from path:{}'.format(init_weight_path))

    def _create_optimizer(self):
        """Creates optimizer."""
        opt = tf.keras.optimizers.Adam(
            self.params["learning_rate"],
            self.params["optimizer_adam_beta1"],
            self.params["optimizer_adam_beta2"],
            epsilon=self.params["optimizer_adam_epsilon"])
        return opt


def _ensure_dir(log_dir):
    """Makes log dir if not existed."""
    if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)


def _build_stats(history, callbacks):
    """Normalizes and returns dictionary of stats."""
    stats = {}

    if history and history.history:
        train_hist = history.history
        # Gets final loss from training.
        stats['loss'] = train_hist['loss'][-1].item()

    if not callbacks:
        return stats

    return stats


def define_transformer_flags():
    flags.DEFINE_string(
        name='mode', default='train',
        help='mode: train, eval or predict')
    flags.DEFINE_integer(
        name='train_steps', default=3000,
        help='The number of steps used to train.')
    flags.DEFINE_string(
        name='data_dir', default='tmp/translate_ende',
        help='The path of data.')
    flags.DEFINE_string(
        name='model_dir', default='tmp/transformer_model',
        help='The path of transformer model.')
    flags.DEFINE_string(
        name='vocab_file', default='tmp/translate_ende/vocab.ende.32768',
        help='Vocab file.')
    flags.DEFINE_integer(
        name='batch_size', default=4096,
        help='Model batch size.')
    flags.DEFINE_integer(
        name='max_length', default=256,
        help='Max sequence length for Transformer.')
    flags.DEFINE_integer(
        name='decode_batch_size', default=32,
        help='Global batch size used for Transformer.')
    flags.DEFINE_integer(
        name='decode_max_length', default=97,
        help='Max sequence length of the decode/eval data.')
    flags.DEFINE_integer(
        name='steps_between_evals', default=1000,
        help='The number of training steps to run between evaluations.')
    flags.DEFINE_boolean(
        name='enable_time_history', default=True,
        help='Whether to enable TimeHistory callback.')
    flags.DEFINE_boolean(
        name='num_parallel_calls', default=True,
        help='None')


def main(_):
    flags_obj = flags.FLAGS
    task = TransformerTask(flags_obj)

    if flags_obj.mode == 'train':
        task.train()
    elif flags_obj.mode == 'predict':
        task.predict()
    elif flags_obj.mode == 'eval':
        task.eval()
    else:
        raise ValueError('Invalid mode {}'.format(flags_obj.mode))


if __name__ == '__main__':
    define_transformer_flags()
    app.run(main)
