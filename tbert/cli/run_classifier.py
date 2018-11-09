from tbert.data import example_to_feats


class BertClassifier(torch.nn.Module):

    def __init__(self, config, num_labels):
        torch.nn.Module.__init__(self, config)

        if config['attention_probs_dropout_prob'] != config['hidden_dropout_prob']:
            raise NotImplementedError()

        dropout = config['attention_probs_dropout_prob']
        hidden_size = config['hidden_size']

        self.bert = Bert(config)

        self.pooler = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

        init_linear(self.pooler, initialization_range)
        init_linear(self.output, initialization_range)

    def forward(self, input_ids, input_type_ids=None, input_mask=None):

        activations = self.bert(input_ids, input_type_ids, input_mask)
        x = self.pooler(activations[-1])
        x = torch.tanh(x)
        x = self.output(x)
        x = self.dropout(x)

        return x


def loss(logits):

    prob = torch.nn.softmax(logits, dim=-1)


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        yield from csv.reader(f, delimiter='\t', quotechar=quotechar)


def _xnli_reader(data_dir, label_vocab, partition='train', lang='zh'):
    if partition == 'train':
        for i, line in enumerate(
                _read_tsv(f'{data_dir}/multinli/multinli.train.{lang}.tsv')
                ):
            if i == 0:
                continue
            guid = f'{partition}-{i}'
            text_a = line[0]
            text_b = line[1]
            label  = line[2]
            if label == 'contradictory':
                label = 'contradiction'
            yield guid, text_a, text_b, label_vocab[label]

    elif partition == 'dev':
        for i, line in enumerate(
                _read_tsv(f'{data_dir}/xnli.dev.tsv')
                ):
            if i == 0:
                continue
            guid = f'{partition}-{i}'
            if line[0] != lang:
                continue
            text_a = line[6]
            text_b = line[7]
            label  = line[1]
            yield guid, text_a, text_b, label_vocab[label]

    else:
        raise ValueError('no such partition in this dataset: %r' % partition)


def _mnli_reader(data_dir, label_vocab, partition='train'):

    fname = {
        'train': f'{data_dir}/train.tsv',
        'dev'  : f'{data_dir}/dev_matched.tsv',
        'test' : f'{data_dir}/test_matched.tsv',
    }.get(partition)

    if fname is None:
        raise ValueError('no such partition in this dataset: %r' % partition)

    for i,line in enumerate(_read_tsv(fname)):
        if i == 0:
            continue
        giud = f'{partition}-{line[0]}'
        text_a = line[8]
        text_b = line[9]
        if partition == 'test':
            label = 'contradiction'
        else:
            label = line[-1]
        yield giud, text_a, text_b, label_vocab[label]


def _mrpc_reader(data_dir, label_vocab, partition='train'):

    fname = {
        'train': f'{data_dir}/train.tsv',
        'dev'  : f'{data_dir}/dev.tsv',
        'test' : f'{data_dir}/test.tsv',
    }.get(partition)

    if fname is None:
        raise ValueError('no such partition in this dataset: %r' % partition)

    for i,line in enumerate(_read_tsv(fname)):
        if i == 0:
            continue
        giud = f'{partition}-{i}'
        text_a = line[8]
        text_b = line[9]
        if partition == 'test':
            label = '0'
        else:
            label = line[-1]
        yield giud, text_a, text_b, label_vocab[label]


def _cola_reader(data_dir, label_vocab, partition='train'):

    fname = {
        'train': f'{data_dir}/train.tsv',
        'dev'  : f'{data_dir}/dev.tsv',
        'test' : f'{data_dir}/test.tsv',
    }.get(partition)

    if fname is None:
        raise ValueError('no such partition in this dataset: %r' % partition)

    for i,line in enumerate(_read_tsv(fname)):
        if partition == 'test' and i == 0:
            continue
        giud = f'{partition}-{i}'
        text_a = line[3]
        text_b = line[4]
        if partition == 'test':
            label = '0'
        else:
            label = line[0]
        yield giud, text_a, text_b, label_vocab[label]


_PROBLEMS = {
    'xnli': dict(
        labels=['contradiction', 'entailment', 'neutral'],
        reader=_xnli_reader
    ),
    'mnli': dict(
        labels=['contradiction', 'entailment', 'neutral'],
        reader=_mnli_reader
    ),
    'mrpc': dict(
        labels=['0', '1'],
        reader=_cola_reader
    ),
    'cola': dict(
        labels=['0', '1'],
        reader=_cola_reader
    ),
}


def labeled_example_to_feats(guid, text_a, text_b, label_id, seq_length, tokenizer):

    feats = example_to_feats(text_a, text_b, seq_length, tokenizer)
    feats.update(label_id=label_id)

    return feats


def feats_reader(reader, seq_length, tokenizer):

    for guid, text_a, text_b, label_id in reader:
        yield labeled_example_to_feats(
            guid,
            text_a,
            text_b,
            label_id,
            seq_length,
            tokenizer
        )


if __name__ == '__main__':
    '''
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/MRPC/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --load_checkpoint $BERT_PYTORCH_DIR/pytorch_model.bin \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/mrpc_output/
    '''
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

    parser.add_argument('--load_checkpoint', help='Directory with pretrained tBERT checkpoint')
    parser.add_argument('--problem', required=True, choices={'cola', 'mnli', 'mrpc', 'xnli'}, help='problem type')
    parser.add_argument('--data_dir', required=True, help='Directory with the data')
    parser.add_argument('--do_train', action='store_true', help='Set this flag to run training')


    args = parser.parse_args()
    if args.command is None:
        parser.error('A valid command is required')

    problem = _PROBLEMS[args.problem]
    label_vocab = {
        label: i
        for i, label in enumerate(problem.labels)
    }
    reader = problem.reader

    ckpt = lambda s: f'{args.load_checkpoint}/{s}'
    out  = lambda s: f'{args.output_dir}/{s}'

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

    if args.load_checkpoint is not None:
        with open(args.load_checkpoint, 'rb') as f:
            bert.load_state_dict(pickle.load(f))

    if args.do_train:
        bert.train()

        reader = feats_reader(
            problem.reader(args.data_dir, 'train'),
            args.max_seq_length,
            tokenizer
        )

        reader = shuffle_stream(reader)


    if args.do_eval:
        bert.eval()

        reader = feats_reader(
            problem.reader(args.data_dir, 'test'),
            args.max_seq_length,
            tokenizer
        )

        with open(f'{args.output_dir}/eval_results.txt', 'w') as f:
            for b in shape_batch(reader, batch_size=args.eval_batch_size):
                input_ids      = torch.LongTensor(b['input_ids'])
                input_type_ids = torch.LongTensor(b['input_type_ids'])
                input_mask     = torch.LongTensor(b['input_mask'])
                label_id       = torch.LongTensor(b['label_id'])

                logits = bert(input_ids, input_type_ids, input_mask, label_id)

    if args.do_predict:
        bert.eval()

        with open(args.output_dir + '/test_results.tsv', 'w') as f:
            for sample in problem.reader(args.data_dir, 'predict'):
                feats = labeled_example_to_feats(*sample, args.max_sequence_len, tokenizer)

                f.write('\t'.join(str(prob) for prob in predictions) + '\n')



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
