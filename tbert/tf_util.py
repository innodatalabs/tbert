# The MIT License
# Copyright 2019 Innodata Labs and Mike Kroutikov
#
'''
Utilities to trace TF graph execution, capture TF BERT parameters,
and convert them to PyTorch.
'''
import contextlib
import tensorflow as tf
import modeling
import torch
import numpy as np
from tbert.bert import Bert, BertPooler


def read_tf_checkpoint(init_checkpoint):
    '''Reads standard TF checkpoint and returns all variables.

    Returns:
        dictionary var_name==>numpy_array
    '''
    c = tf.train.load_checkpoint(init_checkpoint)
    return {
        name: c.get_tensor(name)
        for name in c.get_variable_to_shape_map().keys()
    }

def make_state_dict(vvars, mapping, **fmt):
    '''Creates PyTorch *state dict* from TF variables and mapping info

    Mapping from TF to PyTorch is defined by a dictionary with keys
    being PyTorch parameter name, and values being a dictionary containing:
    * path - the TF variable name
    * transpose - optional True/False flag to indicate that TF value need
        to be transposed when copied to PyTorch

    Path may contain formatting templates, that will be expanded using (optional)
    "fmt" keyword parameters

    Returns PyTorch state dictionary
    '''

    def make_tensor(item):
        var = vvars[item['path'].format(**fmt)]
        if item.get('transpose'):
            var = var.T
        return torch.FloatTensor(var)

    return {
        name: make_tensor(item)
        for name, item in mapping.items()
    }

# Mapping spec for BERT embedder
EMBED_SPEC = {
    'token_embedding.weight' : {
        'path': 'bert/embeddings/word_embeddings',
    },
    'segment_embedding.weight' : {
        'path': 'bert/embeddings/token_type_embeddings',
    },
    'position_embedding' : {
        'path': 'bert/embeddings/position_embeddings',
    },
    'layer_norm.weight' : {
        'path': 'bert/embeddings/LayerNorm/gamma',
    },
    'layer_norm.bias' : {
        'path': 'bert/embeddings/LayerNorm/beta',
    },
}

# mapping spec for BERT encoder
ENCODER_SPEC = {
    'attention.query.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/self/query/kernel',
        'transpose': True
    },
    'attention.query.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/self/query/bias'
    },
    'attention.key.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/self/key/kernel',
        'transpose': True
    },
    'attention.key.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/self/key/bias'
    },
    'attention.value.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/self/value/kernel',
        'transpose': True
    },
    'attention.value.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/self/value/bias'
    },
    'dense.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/output/dense/kernel',
        'transpose': True
    },
    'dense.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/output/dense/bias'
    },
    'dense_layer_norm.weight' : {
        'path': 'bert/encoder/layer_{L}/attention/output/LayerNorm/gamma'
    },
    'dense_layer_norm.bias' : {
        'path': 'bert/encoder/layer_{L}/attention/output/LayerNorm/beta'
    },
    'intermediate.weight' : {
        'path': 'bert/encoder/layer_{L}/intermediate/dense/kernel',
        'transpose': True
    },
    'intermediate.bias' : {
        'path': 'bert/encoder/layer_{L}/intermediate/dense/bias'
    },
    'output.weight' : {
        'path': 'bert/encoder/layer_{L}/output/dense/kernel',
        'transpose': True
    },
    'output.bias' : {
        'path': 'bert/encoder/layer_{L}/output/dense/bias'
    },
    'output_layer_norm.weight' : {
        'path': 'bert/encoder/layer_{L}/output/LayerNorm/gamma'
    },
    'output_layer_norm.bias' : {
        'path': 'bert/encoder/layer_{L}/output/LayerNorm/beta'
    },
}


POOLER_SPEC = {
    'weight' : {
        'path': 'bert/pooler/dense/kernel',
        'transpose': True
    },
    'bias' : {
        'path': 'bert/pooler/dense/bias'
    },
}


def make_bert_state_dict(vvars, num_hidden_layers=12):
    '''Creates tBERT *state dict* from TF BERT parameters'''

    state_dict = {}

    state_dict.update({
        f'embedding.{name}': array
        for name, array in make_state_dict(vvars, EMBED_SPEC).items()
    })

    for layer in range(num_hidden_layers):
        layer_state = make_state_dict(vvars, ENCODER_SPEC, L=layer)
        state_dict.update({
            f'encoder.{layer}.{name}': array
            for name, array in layer_state.items()
        })

    return state_dict


def make_bert_pooler_state_dict(vvars, num_hidden_layers=12):
    '''Creates tBERT *state dict* from TF BERT parameters'''
    state_dict = {}

    state_dict.update({
        f'bert.{name}': array
        for name, array in make_bert_state_dict(vvars, num_hidden_layers).items()
    })

    state_dict.update({
        f'pooler.{name}': array
        for name, array in make_state_dict(vvars, POOLER_SPEC).items()
    })

    return state_dict


class Tracer:
    '''Wraps tf.Session to provide convenience methods to set
    trainable parameters from numpy array, and to read trainable
    parameters as numpy arrays.
    '''

    def __init__(self, sess):
        self.sess = sess
        self._tracer_init_ops = {}

    @property
    def graph(self):
        return self.sess.graph

    def run(self, *av, **kav):
        return self.sess.run(*av, **kav)

    def getmany(self, names):
        graph = self.graph
        vv = [graph.get_tensor_by_name(name+':0') for name in names]

        return dict(zip(names, self.run(vv)))

    def __getitem__(self, name):
        assert type(name_or_names) is str
        return self.getmany([name])[name]

    def update(self, params):
        feed_dict = {}
        to_run = []
        graph = self.graph
        for name, array in params.items():
            if name not in self._tracer_init_ops:
                with tf.name_scope('initTracer'):
                    var = graph.get_tensor_by_name(name+':0')
                    assert array.dtype == np.float32
                    p = tf.placeholder(tf.float32, name=name)
                    op = tf.assign(var, p)
                    self._tracer_init_ops[name] = (op, p)
            op, p = self._tracer_init_ops[name]
            to_run.append(op)
            feed_dict[p] = array
        self.run(to_run, feed_dict=feed_dict)

    def __setitem__(self, name, array):
        self.update({name: array})

    def trainable_variables(self):
        names = [v.name.rstrip(':0') for v in tf.trainable_variables()]
        return self.getmany(names)


@contextlib.contextmanager
def tracer_session():
    with tf.Graph().as_default(), tf.Session() as _sess:
        tracer = Tracer(_sess)
        yield tracer


def run_tf_bert_once(config, params, input_ids, input_type_ids=None, input_mask=None):
    '''Created TF BERT model from config and params, and runs it on the provided inputs

    If initialization params are not provided, then TF BERT is randomly initialized

    Retuns the array of activations of all encoder layers.
    '''
    with tracer_session() as sess:
        if input_type_ids is None:
            input_type_ids = np.zeros_like(input_ids)
        if input_mask is None:
            input_type_ids = np.ones_like(input_ids)

        pinput_ids = tf.placeholder(dtype=tf.int32, shape=input_ids.shape, name='input_ids')
        pinput_mask = tf.placeholder(dtype=tf.int32, shape=input_mask.shape, name='input_mask')
        pinput_type_ids = tf.placeholder(dtype=tf.int32, shape=input_type_ids.shape, name='input_type_ids')

        model = modeling.BertModel(
            modeling.BertConfig.from_dict(config),
            is_training=False,
            input_ids=pinput_ids,
            input_mask=pinput_mask,
            token_type_ids=pinput_type_ids,
            use_one_hot_embeddings=False,
        )

        sess.run(tf.global_variables_initializer())
        sess.update(params)  # set all trainable params

        num_hidden_layers = config['num_hidden_layers']
        to_eval = []
        for reshape_id in range(2, 2 + num_hidden_layers):
            to_eval.append(
                sess.graph.get_tensor_by_name(f'bert/encoder/Reshape_{reshape_id}:0')
            )
        pooler_output = sess.graph.get_tensor_by_name('bert/pooler/dense/Tanh:0')

        out, pout = sess.run((to_eval, pooler_output), feed_dict={
            pinput_ids: input_ids,
            pinput_type_ids: input_type_ids,
            pinput_mask: input_mask
        })

        return out, pout

def get_tf_bert_init_params(config):
    '''Created TF BERT model from config and params, and runs it on the provided inputs

    If initialization params are not provided, then TF BERT is randomly initialized

    Retuns the array of activations of all encoder layers.
    '''
    input_ids = np.zeros(dtype=np.int32, shape=(1, 20))
    input_type_ids = np.zeros_like(input_ids)
    input_mask = np.ones_like(input_ids)

    with tracer_session() as sess:
        pinput_ids = tf.placeholder(dtype=tf.int32, shape=input_ids.shape, name='input_ids')
        pinput_mask = tf.placeholder(dtype=tf.int32, shape=input_mask.shape, name='input_mask')
        pinput_type_ids = tf.placeholder(dtype=tf.int32, shape=input_type_ids.shape, name='input_type_ids')

        model = modeling.BertModel(
            modeling.BertConfig.from_dict(config),
            is_training=False,
            input_ids=pinput_ids,
            input_mask=pinput_mask,
            token_type_ids=pinput_type_ids,
            use_one_hot_embeddings=False,
        )

        sess.run(tf.global_variables_initializer())

        return sess.trainable_variables()


def run_tbert_once(config, params, input_ids, input_type_ids, input_mask):
    '''Runs tBERT model using TF parameters'''

    # init tBERT model
    state_dict = make_bert_state_dict(params, config['num_hidden_layers'])
    bert = Bert(config)
    bert.load_state_dict(state_dict)

    # run tBERT on the same input
    bert.eval()
    with torch.no_grad():
        out = bert(
            torch.LongTensor(input_ids),
            torch.LongTensor(input_type_ids),
            torch.LongTensor(input_mask)
        )

    return [v.data.numpy() for v in out]


def run_tbert_pooler_once(config, params, input_ids, input_type_ids, input_mask):
    '''Runs tBERT model using TF parameters'''

    # init tBERT model
    state_dict = make_bert_pooler_state_dict(params, config['num_hidden_layers'])

    classifier = BertPooler(config)
    classifier.load_state_dict(state_dict)

    # run tBERT on the same input
    classifier.eval()
    with torch.no_grad():
        out = classifier(
            torch.LongTensor(input_ids),
            torch.LongTensor(input_type_ids),
            torch.LongTensor(input_mask)
        )

    return out.data.numpy()
