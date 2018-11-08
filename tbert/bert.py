import torch
from tbert.embedding import BertEmbedding
from tbert.transformer import Transformer


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
            max_position_embeddings=config['max_position_embeddings']
        )

        self.transformer = torch.nn.ModuleList([
            Transformer(
                hidden_size=config['hidden_size'],
                num_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size'],
                dropout=dropout
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
        for layer in self.transformer:
            y = layer(y, att_mask, batch_size=B)
            outputs.append(y)

        return outputs

