import json
from tbert.util import make_state_dict
from tbert.bert import Bert
from extract_features import model_fn_builder
import modeling
import tensorflow as tf


def read_tf_checkpoint(init_checkpoint):
    c = tf.train.load_checkpoint(init_checkpoint)
    return {
        name: c.get_tensor(name)
        for name in tf_checkpoint.get_variable_to_shape_map().keys()
    }

EMBED_SPEC = {
    'token_embedding.weight' : {
        'path': 'bert/embeddings/word_embeddings:0',
    },
    'segment_embedding.weight' : {
        'path': 'bert/embeddings/token_type_embeddings:0',
    },
    'position_embedding' : {
        'path': 'bert/embeddings/position_embeddings:0',
    },
    'layer_norm.weight' : {
        'path': 'bert/embeddings/LayerNorm/gamma:0',
    },
    'layer_norm.bias' : {
        'path': 'bert/embeddings/LayerNorm/beta:0',
    },
}

TRANSFORMER_SPEC = {
    'attention.query.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/self/query/kernel:0',
        'transpose': True
    },
    'attention.query.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/self/query/bias:0'
    },
    'attention.key.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/self/key/kernel:0',
        'transpose': True
    },
    'attention.key.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/self/key/bias:0'
    },
    'attention.value.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/self/value/kernel:0',
        'transpose': True
    },
    'attention.value.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/self/value/bias:0'
    },
    'dense.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/output/dense/kernel:0',
        'transpose': True
    },
    'dense.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/output/dense/bias:0'
    },
    'dense_layer_norm.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/output/LayerNorm/gamma:0'
    },
    'dense_layer_norm.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/output/LayerNorm/beta:0'
    },
    'intermediate.weight' : {
        'path': 'bert/encoder/layer_{L}/intermediate/dense/kernel:0',
        'transpose': True
    },
    'intermediate.bias' : {
        'path': 'bert/encoder/layer_{L}/intermediate/dense/bias:0'
    },
    'output.weight' : {
        'path': 'bert/encoder/layer_{L}/output/dense/kernel:0',
        'transpose': True
    },
    'output.bias' : {
        'path': 'bert/encoder/layer_{L}/output/dense/bias:0'
    },
    'output_layer_norm.weight' : {
        'path': 'bert/encoder/layer_{L}/output/LayerNorm/gamma:0'
    },
    'output_layer_norm.bias' : {
        'path': 'bert/encoder/layer_{L}/output/LayerNorm/beta:0'
    },
}


if __name__ == '__main__':
    import argparse
    import os
    import shutil
    import pickle

    parser = argparse.ArgumentParser(description='Converts TF BERT checkpoint to tBERT one')

    parser.add_argument('input_dir', help='Directory containing pre-trained TF BERT data (bert_config.json, vocab.txt, and bert_model.chpt')
    parser.add_argument('output_dir', help='Directory where to write tBERT cehckoint (will be created if does not exist)')

    args = parser.parse_args()
    if args.input_dir == args.output_dir:
        raise ValueError('Can not write to the same directory as input_dir')

    src = lambda s: args.input_dir + '/' + s
    trg = lambda s: args.output_dir + '/' + s

    with open(src('bert_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(json.dumps(config, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copyfile(src('bert_config.json'), trg('bert_config.json'))
    shutil.copyfile(src('vocab.txt'), trg('vocab.txt'))

    bert_vars = read_tf_checkpoint(src('bert_model.ckpt'))

    bert = Bert(config)

    bert.embedding.load_state_dict(
        make_state_dict(bert_vars, EMBED_SPEC)
    )

    for layer in range(config['num_hidden_layers']):
        bert.transformer[layer].load_state_dict(
            make_state_dict(bert_vars, TRANSFORMER_SPEC, L=layer)
        )

    with open(trg('bert_model.pickle'), 'wb') as f:
        pickle.dump(bert.state_dict(), f)

    print('Sucessfully created tBERT model in', args.output_dir)
