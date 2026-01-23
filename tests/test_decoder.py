import sys
import math
sys.path.append(r"c:\main\GitHub\handwrittenTextRecognitionSystem")

import pytest

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformers import TransformerEncoderLayer, MultiHeadAttention


h = 4
d_model = 256
d_feedforward = 512


class MultiHeadAttentionTest(MultiHeadAttention):
    def __init__(self, d_model:int, h:int):
        super().__init__(d_model, h)
        
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
        k_value = x if x_dec is None else x_dec
        v_value = x if x_dec is None else x_dec
        for head_ind, head in self.heads_dict.items():
            proj_dict[head_ind] = {"q": head["q"](x),
                                   "k": head["k"](x if x_dec is None else x_dec),
                                   "v": head["v"](x if x_dec is None else x_dec)}
        
        # Calculate values with attention for further cancatenation
        score_values = []
        attention_list = []
        for proj in proj_dict.values():
            scores = torch.matmul(proj["q"], proj["k"].transpose(-2, -1))/math.sqrt(self.d_model)

            if mask is not None:
                scores = scores.masked_fill(mask == 0, -torch.inf)

            attention = F.softmax(scores, dim=-1)
            attention_list.append(attention)

            value_scores = torch.matmul(attention, proj["v"])
            score_values.append(value_scores)
        
        # Final linear projection
        output = self.w_o(torch.cat(score_values, dim=-1))

        return attention_list, k_value, v_value


class TransformerDecoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, d_feedforward, h, dropout=0.35,
                 mask:Optional[torch.tensor] = None, x_dec:Optional[torch.tensor] = None):
        super().__init__(d_model, d_feedforward, h, dropout)
        self.mask = mask
        self.x_dec = x_dec
        self.masked_mha = MultiHeadAttention(d_model, h)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x:torch.tensor):
        masked_mha_output = self.masked_mha(x, mask=self.mask)
        masked_mha_output = self.dropout3(masked_mha_output)
        masked_mha_output = self.norm3(x + masked_mha_output)
        # Multi-head attention with residual connection and normalization
        mha_output = self.mha(masked_mha_output, x_dec=self.x_dec)
        mha_output = self.dropout1(mha_output)
        mha_output = self.norm1(masked_mha_output + mha_output)
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(mha_output)
        ff_output = self.dropout2(ff_output) 
        ff_output = self.norm2(mha_output + ff_output)  # Residual connection + layer norm
        return ff_output
    



D_MODEL = 128
BATCH_SIZE = 3
S1, S2 = 20, 32
TEST_NUM = 100

x_tensors = [torch.randn(BATCH_SIZE, S2, D_MODEL) for _ in range(TEST_NUM)]
x_dec_tensors = [torch.randn(BATCH_SIZE, S1, D_MODEL) for _ in range(TEST_NUM)]

@pytest.mark.parametrize('x, x_dec', zip(x_tensors, x_dec_tensors))
def test_decoder_multihead(x: torch.tensor, x_dec: torch.tensor,
                           mask: torch.tensor = torch.triu(torch.ones(S2, S2)).T):
    """
    Tests whether sequence from decoder is masked and sequnce from encoder comes through

    Parameters
        x: torch.tensor
            sequence from decoder pipeline of shape B x S2 x d_model
        x_dec: torch.tensor
            sequence representation from encoder of shape B x S1 x d_model
        mask: torch.tensor
            mask that has zeros above the diagonal
    """
    masked_mha = MultiHeadAttentionTest(d_model=D_MODEL, h=2)
    attention_list, _, _ = masked_mha(x, mask=mask)
    _, k_value, v_value = masked_mha(x, x_dec=x_dec)
    for attention in attention_list:
        assert len(attention.shape) == 3, "Shape is not with batch size"
        for row_ind in range(attention.shape[1]):
            sum_til_diagonal = attention[0, row_ind, :row_ind+1].sum()
            assert abs(sum_til_diagonal.item() - 1.0) < 1e-4
    assert bool((k_value == v_value).all())
    assert bool((v_value == x_dec).all())
'--------------------------------------------------------------------------------'