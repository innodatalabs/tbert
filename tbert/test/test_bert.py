import tensorflow as tf
import torch
import numpy as np
import random
from tbert.tf_util import tracer_session, get_tf_bert_init_params, \
    run_tf_bert_once, run_tbert_once, run_tbert_pooler_once


# to get stable results
tf.set_random_seed(1)
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)


CONFIG_MICRO = dict(
    attention_probs_dropout_prob=0.1,
    directionality="bidi",
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=10,
    initializer_range=0.02,
    intermediate_size=10,
    max_position_embeddings=20,
    num_attention_heads=2,
    num_hidden_layers=1,
    type_vocab_size=2,
    vocab_size=100
)

PARAMS_MICRO = get_tf_bert_init_params(CONFIG_MICRO)


CONFIG_BIG = dict(
    attention_probs_dropout_prob = 0.1,
    directionality = "bidi",
    hidden_act = "gelu",
    hidden_dropout_prob = 0.1,
    hidden_size = 768,
    initializer_range = 0.02,
    intermediate_size = 3072,
    max_position_embeddings = 512,
    num_attention_heads = 12,
    num_hidden_layers = 12,
    pooler_fc_size = 768,
    pooler_num_attention_heads = 12,
    pooler_num_fc_layers = 3,
    pooler_size_per_head = 128,
    pooler_type = "first_token_transform",
    type_vocab_size = 2,
    vocab_size = 105879
)

PARAMS_BIG = get_tf_bert_init_params(CONFIG_BIG)


def assert_same(*av, tolerance=1.e-6):
    tf_out, _ = run_tf_bert_once(*av)
    tbert_out = run_tbert_once(*av)

    # compare
    assert len(tf_out) == len(tbert_out)
    for x,y in zip(tf_out, tbert_out):
        delta = np.max(np.abs(x.flatten()-y.flatten()))
        assert delta < tolerance, delta


def assert_same_pooler(*av, tolerance=1.e-6):
    _, tf_logits = run_tf_bert_once(*av)
    tbert_logits = run_tbert_pooler_once(*av)

    # compare
    delta = np.max(np.abs(tf_logits.flatten()-tbert_logits.flatten()))
    assert delta < tolerance, delta


def make_random_inputs(vocab_size, shape):
    input_ids = np.random.randint(vocab_size, size=shape)
    input_type_ids = np.random.randint(2, size=shape)
    input_mask = np.random.randint(2, size=shape)

    return input_ids, input_type_ids, input_mask


def test_smoke():
    input_ids = np.array([[1, 2, 3, 4, 5, 0]])
    input_type_ids = np.array([[0, 0, 1, 1, 1, 0]])
    input_mask = np.array([[1, 1, 1, 1, 1, 0]])

    assert_same(CONFIG_MICRO, PARAMS_MICRO, input_ids, input_type_ids, input_mask)


def test_random():
    input_ids, input_type_ids, input_mask = make_random_inputs(100, (2, 5))

    assert_same(CONFIG_MICRO, PARAMS_MICRO, input_ids, input_type_ids, input_mask)


def test_random_big():
    input_ids, input_type_ids, input_mask = make_random_inputs(10000, (10, 128))

    assert_same(CONFIG_BIG, PARAMS_BIG, input_ids, input_type_ids, input_mask,
        tolerance=1e-4)


def test_pooler():
    input_ids, input_type_ids, input_mask = make_random_inputs(100, (2, 5))

    assert_same_pooler(CONFIG_MICRO, PARAMS_MICRO, input_ids, input_type_ids, input_mask)


def test_pooler_big():
    input_ids, input_type_ids, input_mask = make_random_inputs(10000, (10, 128))

    assert_same_pooler(CONFIG_BIG, PARAMS_BIG, input_ids, input_type_ids, input_mask,
        tolerance=2.e-6)
