import json
from tbert.tf_util import read_tf_checkpoint, make_bert_pooler_state_dict
from tbert.bert import BertPooler
import modeling
import tensorflow as tf


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

    bert_pooler = BertPooler(config)
    bert_pooler.load_state_dict(
        make_bert_pooler_state_dict(bert_vars, config['num_hidden_layers'])
    )

    bert_pooler.save_pretrained(args.output_dir)

    print('Sucessfully created tBERT model in', args.output_dir)
