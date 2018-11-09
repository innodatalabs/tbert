# tBERT
BERT model converted to PyTorch.

This is a literate port of BERT code from TensorFlow to PyTorch.
See the [original TF BERT repo here](https://github.com/google-research/bert).

There is a script that converts TF BERT pre-trained checkpoint to tBERT: `tbert.cli.convert`

Testing is done to ensure that tBERT code behaves exactly as TF BERT.

See also alternative PyTorch port by guys from HuggingFace: https://github.com/huggingface/pytorch-pretrained-BERT.git

## Installation

Python3 is required.

```
git clone https://github.com/innodatalabs/tbert.git
cd tbert
virtualenv .venv -p python3
. .venv/bin/activate
pip install -r requirements.txt
```

To run unit tests and CLI utilities, original TF BERT code is required (we use it for tokenization):
```
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
```

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

## Fine-tuning a classifier

Download GLUE datasets, as explained here. In the following we assume that
GLUE dataset are in the `glue_data` directory.

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
