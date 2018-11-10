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


class BertEmbedding(torch.nn.Module):

    def __init__(self,
            token_vocab_size=105879,
            segment_vocab_size=2,
            hidden_size=768,
            max_position_embeddings=512,
            initializer_range=0.02):
        '''
        token_vocab_size - size of token (word pieces) vocabulary
        segment_vocab_size - number of segments (BERT uses 2 always, do not change)
        hidden_size - size of the hidden transformer layer (number of embedding dimensions)
        max_position_embeddings - longest sequence size this model will support
        '''
        torch.nn.Module.__init__(self)

        self.token_embedding = torch.nn.Embedding(token_vocab_size, hidden_size, padding_idx=0)
        self.segment_embedding = torch.nn.Embedding(segment_vocab_size, hidden_size)
        self.position_embedding = torch.nn.Parameter(
            data=torch.zeros(
                max_position_embeddings,
                hidden_size,
                dtype=torch.float32
            )
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=1.e-12)

        # apply weight initialization
        torch.nn.init.normal_(self.token_embedding.weight, std=initializer_range)
        torch.nn.init.normal_(self.segment_embedding.weight, std=initializer_range)
        torch.nn.init.normal_(self.position_embedding, std=initializer_range)

    def forward(self, input_ids, input_type_ids):
        '''
        input_ids - LongTensor of shape [B, S] containing token ids (padded with 0)
        input_type_ids - LongTensor of shape [B, S] containing token segment ids.
            These are: 0 for the tokens in first segment, and 1 for the tokens
            in second segment

        Here: B - batch size, S - sequence length
        '''
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        x = self.token_embedding(input_ids)
        s = self.segment_embedding(input_type_ids)
        p = self.position_embedding[:seq_len, :].unsqueeze(0).repeat((batch_size, 1, 1))

        return self.layer_norm(x + s + p)

