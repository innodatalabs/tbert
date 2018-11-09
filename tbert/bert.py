import torch
from tbert.embedding import BertEmbedding
from tbert.transformer import TransformerEncoder
from tbert.attention import init_linear


class Bert(torch.nn.Module):

    def __init__(self, config):
        torch.nn.Module.__init__(self)

        if config['attention_probs_dropout_prob'] != config['hidden_dropout_prob']:
            raise NotImplementedError()

        if config['hidden_act'] != 'gelu':
            raise NotImplementedError()

        dropout = config['attention_probs_dropout_prob']

        self.embedding = BertEmbedding(
            token_vocab_size=config['vocab_size'],
            segment_vocab_size=config['type_vocab_size'],
            hidden_size=config['hidden_size'],
            max_position_embeddings=config['max_position_embeddings'],
            initializer_range=config['initializer_range']
        )

        self.encoder = torch.nn.ModuleList([
            TransformerEncoder(
                hidden_size=config['hidden_size'],
                num_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size'],
                dropout=dropout,
                initializer_range=config['initializer_range']
            )
            for _ in range(config['num_hidden_layers'])
        ])

    def forward(self, input_ids, input_type_ids=None, input_mask=None):
        B = input_ids.size(0)  # batch size

        if input_mask is None:
            input_mask = torch.ones_like(input_ids)
        if input_type_ids is None:
            input_type_ids = torch.zeros_like(input_ids)

        # credit to: https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
        att_mask = input_mask.unsqueeze(1).unsqueeze(2)
        att_mask = (1.0 - att_mask.float()) * -10000.0

        y = self.embedding(input_ids, input_type_ids)

        # reshape to matrix. Apparently for speed -MK
        y = y.view(-1, y.size(-1))

        outputs = []
        for layer in self.encoder:
            y = layer(y, att_mask, batch_size=B)
            outputs.append(y)

        return outputs


class BertPooler(torch.nn.Module):

    def __init__(self, config):
        torch.nn.Module.__init__(self)

        if config['attention_probs_dropout_prob'] != config['hidden_dropout_prob']:
            raise NotImplementedError()

        dropout = config['attention_probs_dropout_prob']
        hidden_size = config['hidden_size']

        self.bert = Bert(config)

        self.pooler = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

        init_linear(self.pooler, config['initializer_range'])

    def forward(self, input_ids, input_type_ids=None, input_mask=None):
        batch_size = input_ids.size(0)

        activations = self.bert(input_ids, input_type_ids, input_mask)

        x = activations[-1]  # use top layer only
        x = x.view(batch_size, -1, x.size(-1))  # [B, S, H]
        # take activations of the first token (aka BERT-style "pooling")
        x = x[:, 0:1, :].squeeze(1)

        x = self.pooler(x)
        x = torch.tanh(x)

        return x
