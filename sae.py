import torch
import torch.nn as nn


class SAE(nn.Module): 
    """
    
    """
    def __init__(self, inp_dim, hidden_dim): 
        super().__init__()
        self.enc = nn.Linear(inp_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dec = nn.Linear(hidden_dim, inp_dim)

    def forward(self, emb): 
        enc = self.enc(emb)
        recon = self.dec(self.act(enc))
        return enc, recon

class TopKSAE(nn.Module): 
    """
    
    """
    def __init__(self, inp_dim, hidden_dim, k): 
        super().__init__()
        self.enc = nn.Linear(inp_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, inp_dim)
        self.k = k
    def forward(self, emb): 
        enc = self.enc(emb)
        recon = self.dec(torch.topk(enc, self.k))
        return enc, recon