import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


'--------------------------------------------------------------------------------'


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_model = d_model
        self.h = h
        
        # Linear projections for query, key, value for each head
        self.heads_dict = nn.ModuleDict({
            f"head_{k}": nn.ModuleDict({"q":nn.Linear(d_model, d_model),
                              "k":nn.Linear(d_model, d_model),
                              "v":nn.Linear(d_model, d_model)}) for k in range(h)
        })
        # Projection of heads concatenation 
        self.w_o = nn.Linear(d_model*h, d_model)
        
    def forward(self, x:torch.tensor, x_dec:Optional[torch.tensor] = None,
                mask:Optional[torch.tensor] = None):
        """
        Arguments
        ---------
        x: torch.tensor
            Embedded sequence of shape BxSxE (B-batch size, S-sequence length, E-embedding dim)
        x_dec: torch.tensor
            Sequence from encoder
        mask: torch.tensor
            Tensor of shape SxS representing casual masking
        """
        assert len(x.shape) == 3, "Shape of tensor has to be BxSxE"
        
        # Calculating projections for each head
        proj_dict = {}
        for head_ind, head in self.heads_dict.items():
            proj_dict[head_ind] = {"q": head["q"](x),
                                   "k": head["k"](x if x_dec is None else x_dec),
                                   "v": head["v"](x if x_dec is None else x_dec)}
        
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
    def __init__(self, d_model:int, d_feedforward:int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_feedforward, h, dropout=0.35):
        super().__init__()
        # Multi-head attention sublayer
        self.mha = MultiHeadAttention(d_model, h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward sublayer
        self.feed_forward = PositionwiseFeedForward(d_model, d_feedforward)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Multi-head attention with residual connection and normalization
        mha_output = self.mha(x)
        mha_output = self.dropout1(mha_output)
        x = self.norm1(x + mha_output)
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output) 
        x = self.norm2(x + ff_output)  # Residual connection + layer norm
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, d_feedforward, 
                 h, num_layers, pad_idx:int = 0, dropout:float = 0.35):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional embeddings (using positional encoding)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_feedforward, h, dropout)
            for _ in range(num_layers)
        ])
        
        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(dropout)
        # Final norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None):
        """
        Args:
            x: torch.tensor
                Input tensor with vocab indecies of shape (batch_size, seq_len)
            mask: Optional[torch.tensor] 
                Optional mask tensor for scores
        
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
        """
        assert len(x.shape) == 2, "Input has to be of shape BxS, B-batch size S-sequence length"
        token_embeds = self.token_embed(x) * math.sqrt(self.d_model)
        pos_embeds = self.pos_embed(torch.arange(0, token_embeds.shape[1], device=token_embeds.device))
        x = token_embeds + pos_embeds
        x = self.embedding_dropout(x)
        
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.norm(x)
        
        return x



class TransformerDecoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, d_feedforward, h, dropout=0.1, mask:Optional[torch.tensor] = None):
        super().__init__(d_model, d_feedforward, h, dropout)
        self.mask = mask
        self.masked_mha = MultiHeadAttention(d_model, h, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        masked_mha_output = self.masked_mha(x, mask=self.mask)
        masked_mha_output = self.dropout3(masked_mha_output)
        masked_mha_output = self.norm3(x + masked_mha_output)
        # Multi-head attention with residual connection and normalization
        mha_output = self.mha(masked_mha_output)
        mha_output = self.dropout1(mha_output)
        mha_output = self.norm1(masked_mha_output + mha_output)
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(mha_output)
        ff_output = self.dropout2(ff_output) 
        ff_output = self.norm2(mha_output + ff_output)  # Residual connection + layer norm
        return ff_output

class TransformerDecoder(TransformerEncoder):
    def __init__(self, vocab_size:int, max_seq_len:int, d_model:int, d_feedforward:int,
                 h:int, num_layers:int, dropout: float = 0.1, mask: bool = True):
        
        super().__init__(vocab_size, max_seq_len, d_model, d_feedforward,
                         h, num_layers, dropout)
        
        self.mask = torch.triu(torch.ones(max_seq_len, max_seq_len)).T if mask else None
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, d_feedforward, h, dropout, mask=self.mask)
            for _ in range(num_layers)
        ])
    




if __name__ == "__main__":
    mha = MultiHeadAttention(256, 4)
    ff = PositionwiseFeedForward(256, 1024)
    trans = TransformerEncoder(vocab_size=10, max_seq_len=20, d_model=256, d_feedforward=1024, h=4, num_layers=3, dropout=0.35)
    x = torch.randint(0, 10, size=(32, 19))
    print(trans(x).shape, end = '\n\n')

    # x = torch.randn(32, 18, 256)
    # decoder_layer = TransformerDecoderLayer(256, 128, 8, dropout=0.15, mask=torch.triu(torch.ones(18, 18)).T)
    # print(decoder_layer(x).shape, end='\n\n')

    # x = torch.randint(0, 10, size=(32, 20))
    # trans_dec = TransformerDecoder(vocab_size=10, max_seq_len=20, d_model=256, d_feedforward=128, h=4, num_layers=3, dropout=0.35)
    # print(trans_dec(x).shape)