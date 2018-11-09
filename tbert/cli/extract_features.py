import json
import pickle
import collections
from tbert.data import parse_example, example_to_feats, batch
import tokenization  # from original BERT repo
import torch


def read_examples(filename, max_seq_len, tokenizer):
    '''Reads examples from text file and converts to features'''

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            text_a, text_b = parse_example(line)
            feats = example_to_feats(
                text_a,
                text_b,
                max_seq_len,
                tokenizer
            )
            yield feats


def predict_json_features(bert, examples, batch_size=32, layer_indexes=None):
    '''Runs BERT model on examples and creates JSON output object for each'''
    if layer_indexes is None:
        layer_indexes = [-1, -2, -3, -4]

    unique_id = 0
    for b in batch(examples, batch_size=batch_size):
        input_ids      = torch.LongTensor(b['input_ids'])
        input_type_ids = torch.LongTensor(b['input_type_ids'])
        input_mask     = torch.LongTensor(b['input_mask'])

        out = bert(input_ids, input_type_ids, input_mask)
        num_items_in_batch = input_ids.size(0)
        for idx in range(num_items_in_batch):
            all_features = []
            output_json = collections.OrderedDict([
                ('linex_index', unique_id),
                ('features', all_features),
            ])
            tokens = b['tokens'][idx]
            for i, tk in enumerate(tokens):
                all_layers = []
                all_features.append(collections.OrderedDict([
                    ('token', tk),
                    ('layers', all_layers)
                ]))
                for j, layer_index in enumerate(layer_indexes):
                    layer_output = out[layer_index]
                    layer_output = layer_output.view(num_items_in_batch, -1, layer_output.size(-1))
                    values = [round(float(x), 6) for x in layer_output[idx, i, :]]
                    all_layers.append(collections.OrderedDict([
                        ('index', layer_index),
                        ('values', values),
                    ]))
            yield output_json
            unique_id += 1


if __name__ == '__main__':
    import argparse
    from tbert.bert import Bert

    parser = argparse.ArgumentParser(description='Reads text file and extracts BERT features for each sample')

    parser.add_argument('input_file', help='Input text file - one example per line')
    parser.add_argument('output_file', help='Name of the output JSONL file')
    parser.add_argument('checkpoint_dir', help='Directory with pretrained tBERT checkpoint')
    parser.add_argument('--layers', default='-1,-2,-3,-4', help='List of layers to include into the output, default="%(default)s"')
    parser.add_argument('--batch_size', default=32, help='Batch size, default %(default)s')
    parser.add_argument('--max_seq_length', default=128, help='Sequence size limit (after tokenization), default is %(default)s')
    parser.add_argument('--do_lower_case', default=True, help='Set to false to retain case-sensitive information, default %(default)s')

    args = parser.parse_args()

    ckpt = lambda s: args.checkpoint_dir + '/' + s

    with open(ckpt('bert_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(json.dumps(config, indent=2))

    if config['max_position_embeddings'] < args.max_seq_length:
        raise ValueError('max_seq_length parameter can not exceed config["max_position_embeddings"]')

    tokenizer = tokenization.FullTokenizer(
        vocab_file=ckpt('vocab.txt'),
        do_lower_case=args.do_lower_case,
    )

    bert = Bert(config)

    with open(ckpt('bert_model.pickle'), 'rb') as f:
        bert.load_state_dict(pickle.load(f))
    bert.eval()

    layer_indexes = eval('[' + args.layers + ']')

    examples = read_examples(args.input_file, args.max_seq_length, tokenizer)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for feat_json in predict_json_features(
                bert,
                examples,
                batch_size=args.batch_size,
                layer_indexes=layer_indexes):
            f.write(json.dumps(feat_json) + '\n')

    print('All done')
