import torch
import torch.nn as nn

class PoemModel(nn.Module):
    def __init__(self, vocab_size, **tranformer_args):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, tranformer_args['d_model'])
        self.tranformer = nn.Transformer(**tranformer_args)
        self.linear = nn.Linear(tranformer_args['d_model'], vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.tranformer(emb) # out = <seq, batch, d_model>
        out = out.permute(1, 0, 2)
        out = self.linear(out)
        return out
