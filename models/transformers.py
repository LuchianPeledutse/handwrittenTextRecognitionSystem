import math

import torch
import torch.nn as nn
import torch.nn.functional as F


'--------------------------------------------------------------------------------'


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_model = d_model
        self.h = h
        
        # Linear projections for query, key, value for each head
        self.heads_list = nn.ModuleDict({
            f"head_{k}": nn.ModuleDict({"q":nn.Linear(d_model, d_model),
                              "k":nn.Linear(d_model, d_model),
                              "v":nn.Linear(d_model, d_model)}) for k in range(h)
        })
        # Projection of heads concatenation and dropout
        self.w_o = nn.Linear(d_model*h, d_model)
        
    def forward(self, x:torch.tensor, mask = None):
        batch_size, seq_len, _ = x.shape
        
        # Calculating projections for each head
        proj_dict = {}
        for head_ind, head in self.heads_list.items():
            proj_dict[head_ind] = {"q": head["q"](x), "k": head["k"](x),
                                   "v": head["v"](x)}
        
        # Calculate values with attention for further cancatenation
        score_values = []
        for proj in proj_dict.values():
            scores = torch.matmul(proj["q"], proj["k"].transpose(-2, -1))/math.sqrt(self.d_model)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -torch.inf)

            attention = F.softmax(scores, dim=-1)

            value_scores = torch.matmul(attention, proj["v"])
            score_values.append(value_scores)
        
        # Final linear projection
        output = self.w_o(torch.cat(score_values, dim=-1))

        return output
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model:int, d_feedforward:int, dropout:float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_feedforward, h, dropout=0.1):
        super().__init__()
        # Multi-head attention sublayer
        self.mha = MultiHeadAttention(d_model, h, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward sublayer
        self.feed_forward = PositionwiseFeedForward(d_model, d_feedforward, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection and normalization
        mha_output= self.mha(x, mask)
        x = self.norm1(x + attn_output)  # Residual connection + layer norm
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)  # Dropout applied before residual
        x = self.norm2(x + ff_output)  # Residual connection + layer norm
        
        return x, attention_weights
    




if __name__ == "__main__":
    mha = MultiHeadAttention(256, 4, 0.1)
    ff = PositionwiseFeedForward(256, 1024, 0.3)
    x = torch.randn(32, 12, 256, requires_grad=True)
    print(x.shape)
    x_mha = mha.forward(x)
    print(x_mha.shape)
    print(ff(x).shape)