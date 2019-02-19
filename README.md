# tBERT
[![PyPI version](https://badge.fury.io/py/tbert.svg)](https://badge.fury.io/py/tbert)
[![Build Status](https://travis-ci.com/innodatalabs/tbert.svg?branch=master)](https://travis-ci.com/innodatalabs/tbert)

BERT model converted to PyTorch.

Please, **do NOT use this repo**, instead use the library from
folks from HuggingFace: https://github.com/huggingface/pytorch-pretrained-BERT.git

This repo is kept as an example of converting TF model to PyTorch (utilis may be handy in case I need
to do some thing like this again).

This is a literal port of BERT code from TensorFlow to PyTorch.
See the [original TF BERT repo here](https://github.com/google-research/bert).

We provide a script to convert TF BERT pre-trained checkpoint to tBERT: `tbert.cli.convert`

Testing is done to ensure that tBERT code behaves exactly as TF BERT.

## License
This work uses MIT license.

Original code is covered by Apache 2.0 License.

## Installation

Python 3.6 or better is required.

Easiest way to install is with the `pip`:
```
pip install tbert
```
Now you can start using tBERT models in your code!

## Pre-trained models
Google-trained models, converted to tBERT format. For description of models, see
the [original TF BERT repo here](https://github.com/google-research/bert#pre-trained-models):

* [Base, Uncased](https://storage.googleapis.com/public.innodatalabs.com/tbert-uncased_L-12_H-768_A-12.zip)
* [Large, Uncased](https://storage.googleapis.com/public.innodatalabs.com/tbert-uncased_L-24_H-1024_A-16.zip)
* [Base, Cased](https://storage.googleapis.com/public.innodatalabs.com/tbert-cased_L-12_H-768_A-12.zip)
* [Large, Cased](https://storage.googleapis.com/public.innodatalabs.com/tbert-cased_L-24_H-1024_A-16.zip)
* [Base, Multilingual Cased (New, recommended)](https://storage.googleapis.com/public.innodatalabs.com/tbert-multi_cased_L-12_H-768_A-12.zip)
* [Base, Multilingual Uncased (Not recommended)](https://storage.googleapis.com/public.innodatalabs.com/tbert-multilingual_L-12_H-768_A-12.zip)
* [Base, Chinese](https://storage.googleapis.com/public.innodatalabs.com/tbert-chinese_L-12_H-768_A-12.zip)

## Using tBERT model in your PyTorch code

### tbert.bert.Bert
This is the main juice - the Bert transformer. It is a normal PyTorch module.
You can use it stand-alone or in combination with other PyTorch modules.

```python
from tbert.bert import Bert

config = dict(
    attention_probs_dropout_prob=0.1,
    directionality="bidi",
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=2,
    vocab_size=105879
)

bert = Bert(config)
# ... should load trained parameters (see below)

input_ids      = torch.LongTensor([[1, 2, 3, 4, 5, 0]])
input_type_ids = torch.LongTensor([[0, 0, 1, 1, 1, 0]])
input_mask     = torch.LongTensor([[1, 1, 1, 1, 1, 0]])

activations = bert(input_ids, input_type_ids, input_mask)
```
Returns an array of activations (for each hidden layer).
Typically only the topmost, or few top layers are used.
Each element in the array is a Tensor of shape [B*S, H]
where B is the batch size, S is the sequence length, and H is the
size of the hidden layer.

### tbert.bert.BertPooler
This is the Bert transformer with pooling layer on the top.
Convenient for sequence classification tasks. Use is very similar to
that of `tbert.bert.Bert` module:
```python
from tbert.bert import Bert

config = dict(
    attention_probs_dropout_prob=0.1,
    directionality="bidi",
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=2,
    vocab_size=105879
)

bert_pooler = BertPooler(config)
# ... should load trained parameters (see below)

input_ids      = torch.LongTensor([[1, 2, 3, 4, 5, 0]])
input_type_ids = torch.LongTensor([[0, 0, 1, 1, 1, 0]])
input_mask     = torch.LongTensor([[1, 1, 1, 1, 1, 0]])

activation = bert_pooler(input_ids, input_type_ids, input_mask)
```
Returns a single tensor of size [B, H], where
B is the batch size, and H is the size of the hidden layer.

### Programmatically loading pre-trained weights
To initialize `tbert.bert.Bert` or `tbert.bert.BertPooler` from pre-trained
saved checkpoint:
```
...
bert = Bert(config)
bert.load_pretrained(dir_name)
```
Here, `dir_name` should be a directory containing pre-trained tBIRT model,
with `bert_model.pickle` and `pooler_model.pickle` files. See below to learn how
to convert published TF BERT pre-trained models to tBERT format.

Similarly, `load_pretrained` method can be used on `tbert.bert.BertPooler`
instance.

## Installing optional dependencies
Optional deps are needed to use CLI utilities:
* to convert TF BERT checkpoint to tBERT format
* to extract features from a sequence
* to run training of a classifier

```
pip install -r requirements.txt
mkdir tf
cd tf
git clone https://github.com/google-research/bert
cd ..
export PYTHONPATH=.:tf/bert
```

Now all is set up:
```
python -m tbert.cli.extract_features --help
python -m tbert.cli.convert --help
python -m tbert.cli.run_classifier --help
```

## Running unit tests
```
pip install pytest
pytest tbert/test
```

## Converting TF BERT pre-trained checkpoint to tBERT

* Download TF BERT checkpoint and unzip it
  ```
  mkdir data
  cd data
  wget https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip
  unzip multilingual_L-12_H-768_A-12.zip
  cd ..
  ```
* Run the converter
  ```
  python -m tbert.cli.convert \
    data/multilingual_L-12_H-768_A-12 \
    data/tbert-multilingual_L-12_H-768_A-12
  ```

## Extracting features

Make sure that you have pre-trained tBERT model (see section above).

```
echo "Who was Jim Henson ? ||| Jim Henson was a puppeteer" > /tmp/input.txt
echo "Empty answer is cool!" >> /tmp/input.txt

python -m tbert.cli.extract_features \
    /tmp/input.txt \
    /tmp/output-tbert.jsonl \
    data/tbert-multilingual_L-12_H-768_A-12
```

## Comparing TF BERT and tBERT results

Run TF BERT `extract_features`:
```
echo "Who was Jim Henson ? ||| Jim Henson was a puppeteer" > /tmp/input.txt
echo "Empty answer is cool!" >> /tmp/input.txt

export BERT_BASE_DIR=data/multilingual_L-12_H-768_A-12

python -m extract_features \
  --input_file=/tmp/input.txt \
  --output_file=/tmp/output-tf.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

This creates file `/tmp/output-tf.jsonl`. Now, compare this to the JSON-L file created
by tBERT:

```
python -m tbert.cli.cmp_jsonl \
    --tolerance 5e-5 \
    /tmp/output-tbert.jsonl \
    /tmp/output-tf.jsonl
```

Expect output similar to this:
```
Max float values delta: 3.6e-05
Structure is identical
```

## Fine-tuning a classifier

Download GLUE datasets, as explained
[here](https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks).
In the following we assume that
GLUE datasets are in the `glue_data` directory.

To train MRPC task, do this:
```
python -m tbert.cli.run_classifier \
    data/tbert-multilingual_L-12_H-768_A-12 \
    /tmp \
    --problem mrpc \
    --data_dir glue_data/MRPC \
    --do_train \
    --num_train_steps 600 \
    --num_warmup_steps 60 \
    --do_eval
```

Expect to see something similar to that:
```
...
Step:        550, loss:  0.039, learning rates: 1.888888888888889e-06
Step:        560, loss:  0.014, learning rates: 1.5185185185185186e-06
Step:        570, loss:  0.017, learning rates: 1.1481481481481482e-06
Step:        580, loss:  0.021, learning rates: 7.777777777777779e-07
Step:        590, loss:  0.053, learning rates: 4.074074074074075e-07
Step:        600, loss:  0.061, learning rates: 3.703703703703704e-08
Saved trained model
*** Evaluating ***
Number of samples evaluated: 408
Average per-sample loss: 0.4922609218195373
Accuracy: 0.8504901960784313
```
