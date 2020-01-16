"""Translate text or files using trained transformer model."""

import numpy as np
import tensorflow as tf
import logging
from utils import tokenizer

_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6


def _trim_and_decode(ids, subtokenizer):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
        index = list(ids).index(tokenizer.EOS_ID)
        return subtokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
        return subtokenizer.decode(ids)


def translate_from_input(outputs, subtokenizer):
    translation = _trim_and_decode(outputs, subtokenizer)
    logging.info("Translation: \"%s\"" % translation)
