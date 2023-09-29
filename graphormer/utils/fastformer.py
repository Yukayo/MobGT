import einops
from einops import rearrange
import torch
import torch.nn as nn


class FastformerAttention(nn.Module):
    def __init__(self, dim=3, decode_dim=16):
        super(FastformerAttention, self).__init__()
        # Generate weight for Wquery、Wkey and Wvalue
        self.to_qkv = nn.Linear(dim, decode_dim * 3, bias=False)
        self.weight_q = nn.Linear(dim, decode_dim, bias=False)
        self.weight_k = nn.Linear(dim, decode_dim, bias=False)
        self.weight_v = nn.Linear(dim, decode_dim, bias=False)
        self.weight_r = nn.Linear(decode_dim, decode_dim, bias=False)
        self.weight_alpha = nn.Parameter(torch.randn(decode_dim))
        self.weight_beta = nn.Parameter(torch.randn(decode_dim))
        self.scale_factor = decode_dim ** -0.5

    def forward(self, x, attn_bias=None, mask=None):
        query = self.weight_q(x)
        key = self.weight_k(x)
        value = self.weight_v(x)
        b, n, d = query.shape

        # mask_value = torch.finfo(x.dtype).min
        # mask = rearrange(mask, "b n -> b () n")

        # Caculate the global query
        alpha_weight = torch.softmax(
            torch.mul(query, self.weight_alpha) * self.scale_factor, dim=-1
        )
        global_query = query * alpha_weight
        # global_query = global_query.masked_fill(~mask, mask_value)
        global_query = torch.einsum("b n d -> b d", global_query)

        # Model the interaction between global query vector and the key vector
        repeat_global_query = einops.repeat(global_query, "b d -> b copy d", copy=n)
        # attn_bias = attn_bias.sum(dim=-1).permute(0,2,1)
        # if attn_bias is not None:
        #     p = (repeat_global_query + attn_bias) * key
        # else:
        p = repeat_global_query * key
        beta_weight = torch.softmax(
            torch.mul(p, self.weight_beta) * self.scale_factor, dim=-1
        )
        global_key = p * beta_weight
        global_key = torch.einsum("b n d -> b d", global_key)

        # key-value
        key_value_interaction = torch.einsum("b j, b n j -> b n j", global_key, value)
        key_value_interaction_out = self.weight_r(key_value_interaction)
        result = key_value_interaction_out + query
        return result
