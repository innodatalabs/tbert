# The MIT License
# Copyright 2019 Innodata Labs and Mike Kroutikov
#
# PyTorch port of
# https://github.com/google-research/bert/modeling.py
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
import torch
import torch.nn.functional as F
import math


def init_linear(m, initializer_range):
    torch.nn.init.normal_(m.weight, std=initializer_range)
    m.bias.data.zero_()


class Attention(torch.nn.Module):

    def __init__(self,
            query_size,
            key_size,
            num_heads,
            head_size,
            dropout=0.1,
            initializer_range=0.02
        ):
        torch.nn.Module.__init__(self)

        self.query_size = query_size
        self.key_size = key_size
        self.num_heads = num_heads
        self.head_size = head_size

        self.query = torch.nn.Linear(query_size, key_size)
        self.key   = torch.nn.Linear(key_size, key_size)
        self.value = torch.nn.Linear(key_size, key_size)
        self.dropout = torch.nn.Dropout(dropout)

        init_linear(self.query, initializer_range)
        init_linear(self.key, initializer_range)
        init_linear(self.value, initializer_range)

    def forward(self, query, key, value, mask=None, batch_size=1):
        '''
        query [B*Q, N*H] - query sequence
        key   [B*K, N*H] - key sequence
        value [B*K, N*H] - value sequence
        mask  [B, 1, Q, K] - attention mask (optional)
        batch_size - the batch size (for attention reshaping)

        where:
        B - batch size
        Q - sequence length of query
        K - sequence length of key and value (must be the same)
        N - number of heads
        H - size of one head

        returns:
        [B*K, N*H] - value weighted with the attention
        '''
        B = batch_size
        Q = query.size(0) // batch_size
        K = key.size(0) // batch_size
        N = self.num_heads
        H = self.head_size

        q = self.query(query)  # [B*Q, N*H]
        k = self.key(key)      # [B*K, N*H]
        v = self.value(value)  # [B*K, N*H]

        # [B*Q, N*H] -> [B, Q, N, H] -> [B, N, Q, H]
        q = q.view(B, Q, N, H).transpose(1, 2)
        # [B*K, N*H] -> [B, K, N, H] -> [B, N, K, H]
        k = k.view(B, K, N, H).transpose(1, 2)

        # -> [B, N, Q, K]
        scores = torch.matmul(q, k.transpose(2, 3))
        scores *= 1. / math.sqrt(H)

        if mask is not None:
            scores += mask

        w = F.softmax(scores, dim=3)
        w = self.dropout(w)

        # [B*K, N*H] -> [B, K, N, H] -> [B, N, K, H]
        v = v.view(B, K, N, H).transpose(1, 2)

        # [B, N, Q, H] -> [B, Q, N, H]
        c = torch.matmul(w, v).transpose(1, 2).contiguous()

        # [B*Q, N*H]
        return c.view(-1, N * H)

