import warnings
from typing import Callable, List, Optional, Tuple, Union
import sys

import random

import torch
from torch import nn
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn import functional as F
from clip.gumbel import gumbel_softmax


def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    E = q.size(-1)
    if k is v:
        if q is k:
            proj = F.linear(q, w, b)
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = F.linear(q, w_q, b_q)
            kv_proj = F.linear(k, w_kv, b_kv)
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


def multi_head_attention_forward(
    text_layer: bool,
    idx: int,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    need_weights: bool = False,
    attn_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    prefix: Tensor = None,
    batch_weight: Tensor = None,
    gumbel_layer: Optional[nn.Linear] = None,
    temperature: float = 3.0,
    ori_image_features: Tensor = None
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if prefix is not None:
        _, _, prefix_len, _ = prefix.shape

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias) 
    if prefix is not None:
        prefix_k = prefix[:, 0, ...].transpose(0, 1).contiguous()
        prefix_v = prefix[:, 1, ...].transpose(0, 1).contiguous()

    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=q.dtype,
        check_other=False,
    )

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
    
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    if prefix is not None:
        prefix_k = prefix_k.view(prefix_k.size(0), bsz * num_heads, head_dim).transpose(0, 1)
        prefix_v = prefix_v.view(prefix_v.size(0), bsz * num_heads, head_dim).transpose(0, 1)
    if attn_mask is not None:
        if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    q = q.view(bsz, num_heads, tgt_len, head_dim)

    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)
    if prefix is not None:
        prefix_k = prefix_k.view(bsz, num_heads, prefix_len, head_dim)
        prefix_v = prefix_v.view(bsz, num_heads, prefix_len, head_dim)

    attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal) 
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)  
    if prefix is not None:
        attn_output_prefix = F.scaled_dot_product_attention(q, prefix_k, prefix_v, None, dropout_p, is_causal)  
        if batch_weight is not None:
            attn_output_prefix = attn_output_prefix * batch_weight.view(bsz, 1, 1, 1)
        attn_output_prefix = attn_output_prefix.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)  
        
        if not text_layer:
            gumbel_logits = gumbel_layer(ori_image_features)
            gumbel = gumbel_softmax(gumbel_logits, temperature=temperature)     
            attn_output = (attn_output * gumbel[:,:1].repeat_interleave(tgt_len, dim=0) + \
                        (attn_output + attn_output_prefix) * gumbel[:,1:].repeat_interleave(tgt_len, dim=0)).to(attn_output.dtype)
        else:
            attn_output += attn_output_prefix   
    
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))  
    return attn_output, None

class CustomNN(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(CustomNN, self).__init__()
        self.gumbel_fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, data):
        data = self.gumbel_fc(data)
        return data

class MultiNN(nn.Module):
    def __init__(self, input_dim, num_networks=11, output_dim=2):
        super(MultiNN, self).__init__()
        self.networks = nn.ModuleList([
            CustomNN(input_dim, output_dim) for _ in range(num_networks)
        ])
        self._initialize_weights()
        
    def _initialize_weights(self):
        for network in self.networks:
            for layer in network.modules():
                if isinstance(layer, nn.Linear):
                    xavier_uniform_(layer.weight)
                    constant_(layer.bias, 0.)

    def __getitem__(self, index):
        assert 0 <= index < len(self.networks), "Index 超出范围！"
        return self.networks[index]


class MultiheadAttention(Module): 
    
    def __init__(self, embed_dim, num_heads, device=None, dtype=None, prefix_pool_size=0, prefix_len=0, design_details=None, text_layer=False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.text_layer = text_layer
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)) 
        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)
        
        xavier_uniform_(self.in_proj_weight)
        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

        self.add_preifx = False
        if prefix_pool_size > 0:   
            if not text_layer:
                self.gumbel_fc = MultiNN(input_dim=512, num_networks=prefix_pool_size)
            
            self.add_preifx = True
            prefix_shape = (prefix_pool_size, 2, prefix_len, embed_dim)
            prefix_pool = torch.zeros(prefix_shape, dtype=torch.float16)
            torch.nn.init.uniform_(prefix_pool[:, 0], -1, 1) 
            self.prefix_pool = Parameter(prefix_pool)            
              
    def forward(
            self,
            idx: int,
            query: Tensor,
            key: Tensor, 
            value: Tensor,
            need_weights: bool = False,
            attn_mask: Optional[Tensor] = None,
            prompt_ids: Tensor = None,
            batch_weight: Tensor = None,
            ori_image_features: Tensor = None
            ) -> Tuple[Tensor, Optional[Tensor]]:
        

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        prefix = None
        gumbel_layer = None
        
        if self.add_preifx:
            assert prompt_ids.size(1) == 1, "Only single prefix for one sample is supported."
            prompt_ids = prompt_ids.squeeze(1)
            if prompt_ids.size(0) == 1:
                if not self.text_layer:
                    gumbel_layer = self.gumbel_fc[prompt_ids[0]]                
                prompt_ids = prompt_ids.repeat(query.size(1))
            prefix = self.prefix_pool[prompt_ids]    
        attn_output, attn_output_weights = multi_head_attention_forward(
            self.text_layer, idx, query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            None, None, False,
            0., self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            need_weights=need_weights,
            attn_mask=attn_mask, 
            prefix=prefix, 
            batch_weight=batch_weight,
            gumbel_layer=gumbel_layer,
            ori_image_features=ori_image_features)
        return attn_output, attn_output_weights 