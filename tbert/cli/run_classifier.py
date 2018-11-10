# The MIT License
# Copyright 2019 Innodata Labs and Mike Kroutikov
#
# PyTorch port of
# https://github.com/google-research/bert/run_classifier.py
#
# Original code copyright follows:
#
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import json
#
import pickle
import json
import csv
import itertools
import torch
from torch.utils import data
import torch.nn.functional as F
from tbert.data import example_to_feats
from tbert.bert import BertPooler
from tbert.attention import init_linear
from tbert.data import repeating_reader, batcher, shuffler
from tbert.optimization import LinearDecayWithWarpupLR
import tokenization


class BertClassifier(torch.nn.Module):

    def __init__(self, config, num_classes):
        torch.nn.Module.__init__(self)

        self.bert_pooler = BertPooler(config)
        self.output = torch.nn.Linear(config['hidden_size'], num_classes)
        self.dropout = torch.nn.Dropout(config['hidden_dropout_prob'])

        init_linear(self.output, config['initializer_range'])

    def forward(self, input_ids, input_type_ids=None, input_mask=None):

        x = self.bert_pooler(input_ids, input_type_ids, input_mask)
        x = self.output(x)
        x = self.dropout(x)

        return x

    def load_pretrained(self, dir_name):
        self.bert_pooler.load_pretrained(dir_name)


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
        text_a = line[3]
        text_b = line[4]
        if partition == 'test':
            label = '0'
        else:
            label = line[0]
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
        if partition == 'test':
            label = '0'
            text_a = line[1]
        else:
            label = line[1]
            text_a = line[3]
        yield giud, text_a, None, label_vocab[label]


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
        reader=_mrpc_reader
    ),
    'cola': dict(
        labels=['0', '1'],
        reader=_cola_reader
    ),
}


def feats_reader(reader, seq_length, tokenizer):
    '''Reads samples from reader and makes a feature dictionary for each'''

    for guid, text_a, text_b, label_id in reader:
        feats = example_to_feats(text_a, text_b, seq_length, tokenizer)
        feats.update(label_id=label_id, guid=guid)
        yield feats


if __name__ == '__main__':
    import argparse
    from tbert.bert import Bert

    parser = argparse.ArgumentParser(description='Reads text file and extracts BERT features for each sample')

    parser.add_argument('pretrained_dir', help='Directory with pretrained tBERT checkpoint')
    parser.add_argument('output_dir', help='Where to save trained model (and were to load from for evaluation/prediction)')
    parser.add_argument('--batch_size', default=32, help='Batch size, default %(default)s')
    parser.add_argument('--max_seq_length', default=128, help='Sequence size limit (after tokenization), default is %(default)s')
    parser.add_argument('--do_lower_case', default=True, help='Set to false to retain case-sensitive information, default %(default)s')

    parser.add_argument('--problem', required=True, choices={'cola', 'mnli', 'mrpc', 'xnli'}, help='problem type')
    parser.add_argument('--data_dir', required=True, help='Directory with the data')
    parser.add_argument('--do_train', action='store_true', help='Set this flag to run training')
    parser.add_argument('--do_eval', action='store_true', help='Set this flag to run evaluation')
    parser.add_argument('--do_predict', action='store_true', help='Set this flag to run prediction')

    parser.add_argument('--learning_rate', default=2.e-5, help='Learning rate for training, default %(default)s')
    parser.add_argument('--num_train_steps', default=1000, type=int, help='Number of training steps, default %(default)s')
    parser.add_argument('--num_warmup_steps', default=200, type=int, help='Number of learning rate warmup steps, default %(default)s')
    parser.add_argument('--macro_batch', default=1, help='Number of batches to accumulate gradiends before optimizer does the update, default %(default)s')
    parser.add_argument('--print_every', default=10, help='How often to print training stats, default %(default)s')

    args = parser.parse_args()

    problem = _PROBLEMS[args.problem]
    label_vocab = {
        label: i
        for i, label in enumerate(problem['labels'])
    }
    problem_reader = problem['reader']

    inp = lambda s: f'{args.pretrained_dir}/{s}'
    out = lambda s: f'{args.output_dir}/{s}'

    with open(inp('bert_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(json.dumps(config, indent=2))

    if config['max_position_embeddings'] < args.max_seq_length:
        raise ValueError('max_seq_length parameter can not exceed config["max_position_embeddings"]')

    print('Loading vocabulary...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=inp('vocab.txt'),
        do_lower_case=args.do_lower_case,
    )
    print('Done loading vocabulary.')

    classifier = BertClassifier(config, len(label_vocab))
    if args.do_train:
        print('Loading pre-trained weights...')
        classifier.load_pretrained(args.pretrained_dir)
        print('Done loading pre-trained weights.')

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    classifier.to(device)

    if args.do_train:
        print('*** Training ***')
        classifier.train()

        reader = repeating_reader(
            -1, # repeat indefinetely
            problem_reader,
            args.data_dir,
            label_vocab,
            partition='train'
        )

        reader = shuffler(reader, buffer_size=1000)

        reader = feats_reader(
            reader,
            args.max_seq_length,
            tokenizer
        )

        opt = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1.e-6
        )

        lr_schedule = LinearDecayWithWarpupLR(
            opt,
            args.num_train_steps,
            args.num_warmup_steps
        )

        step = 0
        for b in itertools.islice(
                batcher(reader, batch_size=args.batch_size),
                args.num_train_steps*args.macro_batch):
            input_ids      = torch.LongTensor(b['input_ids']).to(device)
            input_type_ids = torch.LongTensor(b['input_type_ids']).to(device)
            input_mask     = torch.LongTensor(b['input_mask']).to(device)
            label_id       = torch.LongTensor(b['label_id']).to(device)

            logits = classifier(input_ids, input_type_ids, input_mask)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_probs, label_id, reduction='elementwise_mean')
            loss.backward()

            step += 1
            if step % args.macro_batch == 0:
                opt.step()
                lr_schedule.step()
                opt.zero_grad()

            if step % args.print_every == 0:
                lrs = [p['lr'] for p in opt.param_groups][0]
                print(f'Step: {step:>10}, loss: {loss.item():6.2}, learning rates: {lrs:8}')

        # save trained
        with open(f'{args.output_dir}/bert_classifier.pickle', 'wb') as f:
            pickle.dump(classifier.state_dict(), f)
            print('Saved trained model')
    else:
        # load trained
        with open(f'{args.output_dir}/bert_classifier.pickle', 'rb') as f:
            classifier.load_state_dict(pickle.load(f))
        print('Loaded checkpoint')

    if args.do_eval:
        print('*** Evaluating ***')
        classifier.eval()

        reader = feats_reader(
            problem_reader(args.data_dir, label_vocab, partition='dev'),
            args.max_seq_length,
            tokenizer
        )

        total_loss = 0.
        total_samples = 0
        total_hits = 0
        for b in batcher(reader, batch_size=args.batch_size):
            input_ids      = torch.LongTensor(b['input_ids']).to(device)
            input_type_ids = torch.LongTensor(b['input_type_ids']).to(device)
            input_mask     = torch.LongTensor(b['input_mask']).to(device)
            label_id       = torch.LongTensor(b['label_id']).to(device)

            logits = classifier(input_ids, input_type_ids, input_mask)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(log_probs, label_id, reduction='sum').item()
            prediction = torch.argmax(log_probs, dim=-1)
            hits = (label_id == prediction).sum().item()

            total_loss += loss
            total_hits += hits
            total_samples += input_ids.size(0)

        print('Number of samples evaluated:', total_samples)
        print('Average per-sample loss:', total_loss / total_samples)
        print('Accuracy:', total_hits / total_samples)

    if args.do_predict:
        print('*** Predicting ***')
        classifier.eval()

        reader = feats_reader(
            problem_reader(args.data_dir, label_vocab, partition='test'),
            args.max_seq_length,
            tokenizer
        )

        with open(f'{args.output_dir}/test_results.tsv', 'w') as f:
            for b in batcher(reader, batch_size=args.batch_size):
                input_ids      = torch.LongTensor(b['input_ids']).to(device)
                input_type_ids = torch.LongTensor(b['input_type_ids']).to(device)
                input_mask     = torch.LongTensor(b['input_mask']).to(device)

                logits = classifier(input_ids, input_type_ids, input_mask)
                prob = F.softmax(logits, dim=-1)
                for i in range(prob.size(0)):
                    f.write('\t'.join(str(p) for p in prob[i].tolist()) + '\n')

    print('All done')
