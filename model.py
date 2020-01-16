import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class CBOWModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, emb_size):
        super(CBOWModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, emb_size)
        self.decoder = nn.Linear(emb_size, ntoken)


    def forward(self, input):
        emb = self.encoder(input)
        hidden = torch.mean(emb, axis=1)
        #print("hidden:\n", hidden)
        #print("hidden shape:\n", hidden.shape)
        decoded = self.decoder(hidden)
        
        return decoded

