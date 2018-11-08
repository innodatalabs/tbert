# tBERT
BERT model converted to PyTorch.

This is a literate line-by-line port of BERT code from TensorFlow to PyTorch.
See the [original TF BERT repo here](https://github.com/google-research/bert).

There is a script that converts TF BERT pre-trained checkpoint to tBERT: `tbert.cli.convert`

Testing is done to ensure that tBERT code behaves exactly as TF BERT.

See also alternative PyTorch port by guys from HuggingFace: https://github.com/huggingface/pytorch-pretrained-BERT.git

## Installation

```
virtualenv .venv -p python3
. .venv/bin/activate
pip install tbert
```

To run unit tests and CLI utilities, original TF BERT code is required, as well as
TensorFlow library:

```
pip install tensorflow
mkdir tf
cd tf
git clone https://github.com/google-research/bert
cd ..
export PYTHONPATH=.:tf/bert
python -m tbert.cli.extract_features --help
python -m tbert.cli.convert --help
```

Original TF BERT code is needed as we use it to do the tokenization.

## Running unit tests

```
pip install pytest
pytest tbert/test
```

### Converting TF BERT pre-trained checkpoint to tBERT

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

### Extracting features

Make sure that you have pre-trained tBERT model (see section above).

```
echo "Who was Jim Henson ? ||| Jim Henson was a puppeteer" > /tmp/input.txt
echo "Empty answer is cool!" >> /tmp/input.txt

python -m tbert.cli.extract_features \
    /tmp/input.txt \
    /tmp/output-tbert.jsonl \
    data/tbert-multilingual_L-12_H-768_A-12
```

### Comparing TF BERT and tBERT results

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
