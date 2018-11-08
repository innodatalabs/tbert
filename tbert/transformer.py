import torch
from tbert.gelu import gelu
from tbert.attention import Attention


class Transformer(torch.nn.Module):

    def __init__(self,
            hidden_size=768,
            num_heads=12,
            intermediate_size=3072,
            dropout=0.1):
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
            dropout=dropout
        )

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.dense_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1.e-12)
        self.intermediate = torch.nn.Linear(hidden_size, intermediate_size)
        self.output = torch.nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1.e-12)

    def forward(self, inp, att_mask, batch_size=1):
        '''
        B - batch size
        S - sequence length
        H - hidden size

        inp - a float matrix with embedded input sequences, shape [B*S, H]
        att_mask - an int tensor of shape [B, S, S] defining attention mask
        batch_size - batch size

        Returns: a matrix of the same dims as inp (so that transformars are 
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
