# The MIT License
# Copyright 2019 Innodata Labs and Mike Kroutikov
#
# Heavily borrows from:
# https://github.com/google-research/bert/extract_features.py
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
import re
import collections
import random
import torch


def parse_example(line):
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        return (line, None)

    return m.group(1), m.group(2)


def example_to_feats(text_a, text_b, seq_length, tokenizer):

    tokens_a = tokenizer.tokenize(text_a)

    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ['[CLS]']
    tokens.extend(tokens_a)
    tokens.append('[SEP]')

    input_type_ids = [0] * len(tokens)

    if tokens_b:
        tokens.extend(tokens_b)
        tokens.append('[SEP]')
        input_type_ids.extend([1]*(len(tokens_b)+1))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < seq_length:
        padding = [0] * (seq_length - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        input_type_ids.extend(padding)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return dict(
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids
    )


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def group(sequence, batch_size=32, allow_incomplete=True):
    '''Groups input stream into batches of at most batch_size'''
    buffer = []
    for s in sequence:
        buffer.append(s)
        if len(buffer) >= batch_size:
            yield buffer[:]
            buffer.clear()

    if len(buffer) > 0 and allow_incomplete:
        yield buffer


def batcher(sequence, batch_size=32, allow_incomplete=True):
    '''Batches input sequence of features'''

    def shape_batch(batch):
        out = collections.defaultdict(list)
        for seq in batch:
            for key, val in seq.items():
                out[key].append(val)
        return dict(out)

    for batch in group(
            sequence,
            batch_size=batch_size,
            allow_incomplete=allow_incomplete
        ):
        yield shape_batch(batch)


def shuffler(stream, buffer_size=100000):
    '''Shuffles stream of input samples.

    Uses internal buffer to hold samples delayed for shuffling.
    Bigger buffer size gives better shuffling.
    '''
    buffer = []
    for sample in stream:
        if len(buffer) >= buffer_size:
            random.shuffle(buffer)
            yield from buffer[len(buffer)//2:]
            del buffer[len(buffer)//2:]
        buffer.append(sample)

    random.shuffle(buffer)
    yield from buffer


def repeating_reader(num_epochs: int, reader_factory, *av, **kav):
    '''Creates a bigger stream of samples by repeating data
    for the specified number of epochs. To repeat indefinetely
    use num_epochs=-1
    '''
    while num_epochs != 0:
        yield from reader_factory(*av, **kav)
        num_epochs -= 1
