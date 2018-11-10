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
from tbert.gelu import gelu
from tbert.attention import Attention, init_linear


class TransformerEncoder(torch.nn.Module):

    def __init__(self,
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            dropout=0.1,
            initializer_range=0.02):
        '''
        hidden_size - hidden size, must be multiple of num_heads
        num_heads - number of attention heads.
        intermediate_size - size of the intermediate dense layer
        dropout - dropout probability (0. means "no dropout")
        initializer_range - stddev for random weight matrix initialization
        '''
        torch.nn.Module.__init__(self)

        if hidden_size % num_heads:
            raise ValueError(
                'hidden size must be a multiple of the number of attention heads'
            )

        self.attention = Attention(
            hidden_size,
            hidden_size,
            num_heads,
            hidden_size // num_heads,
            dropout=dropout,
            initializer_range=initializer_range
        )

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.dense_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1.e-12)
        self.intermediate = torch.nn.Linear(hidden_size, intermediate_size)
        self.output = torch.nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1.e-12)

        init_linear(self.dense, initializer_range)
        init_linear(self.intermediate, initializer_range)
        init_linear(self.output, initializer_range)

    def forward(self, inp, att_mask=None, batch_size=1):
        '''
        B - batch size
        S - sequence length
        H - hidden size

        inp - a float matrix with embedded input sequences, shape [B*S, H]
        att_mask - an int tensor of shape [B, 1, S, S] - the self-attention mask
        batch_size - batch size

        Returns: a matrix of the same dims as inp (so that encoders are
            stackable)
        '''
        # --> [B*S, H]
        x = self.attention(inp, inp, inp, att_mask, batch_size=batch_size)
        # --> [B*S, H]
        x = self.dense(x)
        x = self.dropout(x)
        x = self.dense_layer_norm(inp + x)
        x2 = self.output(gelu(self.intermediate(x)))
        x = self.output_layer_norm(x + x2)

        return x


class TransformerDecoder(torch.nn.Module):

    def __init__(self,
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            dropout=0.1,
            initializer_range=0.02):
        '''
        hidden_size - hidden size, must be multiple of num_heads
        num_heads - number of attention heads.
        intermediate_size - size of the intermediate dense layer
        dropout - dropout probability (0. means "no dropout")
        '''
        torch.nn.Module.__init__(self)

        if hidden_size % num_heads:
            raise ValueError(
                'hidden size must be a multiple of the number of attention heads'
            )

        self.attention = Attention(
            hidden_size,
            hidden_size,
            num_heads,
            hidden_size // num_heads,
            dropout=dropout,
            initializer_range=initializer_range
        )

        self.encoder_attention = Attention(
            hidden_size,
            hidden_size,
            num_heads,
            hidden_size // num_heads,
            dropout=dropout,
            initializer_range=initializer_range
        )

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.dense_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1.e-12)
        self.intermediate = torch.nn.Linear(hidden_size, intermediate_size)
        self.output = torch.nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1.e-12)

        init_linear(self.dense, initializer_range)
        init_linear(self.intermediate, initializer_range)
        init_linear(self.output, initializer_range)

    def forward(self, inp, enc_inp, att_mask=None, enc_att_mask=None, batch_size=1):
        '''
        B - batch size
        S - sequence length
        E - encoder sequence length
        H - hidden size

        inp - a float matrix with embedded input sequences, shape [B*S, H]
        enc_inp - a float matrix with embedded activations from encoder layer, shape [B*E, H]
        att_mask - an int tensor of shape [B, 1, S, S] - the self-attention mask
        enc_att_mask - an int tensor of shape [B, 1, E, S] - the attention mask from encoder data
        batch_size - batch size

        Returns: a matrix of the same dims as inp (so that decoders are
            stackable)
        '''
        # --> [B*S, H]
        x = self.attention(inp, inp, inp, att_mask, batch_size=batch_size)

        # apply attention on encoder
        x = self.encoder_attention(enc_inp, x, x, enc_att_mask, batch_size=batch_size)

        # --> [B*S, H]
        x = self.dense(x)
        x = self.dropout(x)
        x = self.dense_layer_norm(inp + x)
        x2 = self.output(gelu(self.intermediate(x)))
        x = self.output_layer_norm(x + x2)

        return x
