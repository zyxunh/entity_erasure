import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers import UNet2DConditionModel
import kornia


class CustomAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, attention_collector=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.name = name
        self.attention_collector = attention_collector

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        is_cross_attention = encoder_hidden_states is not None
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            print(deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if not is_cross_attention:
            masked_restrict = self.attention_collector and "inpainting_mask" in self.attention_collector and self.name in \
                             self.attention_collector['inpainting_mask_used_layers'] and self.attention_collector[
                                 'timestep'] > \
                             self.attention_collector['masked_restrict_step']
            if masked_restrict:
                map_hw = [int(key.shape[-2] ** 0.5)] * 2
                inpainting_mask = F.interpolate(self.attention_collector["inpainting_mask"], size=map_hw, mode="bilinear")
                attention_mask = (inpainting_mask < 0.25).to(inpainting_mask)
                for _ in range(1):
                    attention_mask = kornia.morphology.erosion(attention_mask, kernel=inpainting_mask.new_ones([3, 3]))
                attention_mask = attention_mask.flatten(-2).unsqueeze(-2).repeat(batch_size, attn.heads, query.shape[2], 1).bool()

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if not is_cross_attention:
            if len(self.attention_collector):
                with torch.no_grad():
                    scale_factor = 1 / math.sqrt(query.size(-1))
                    attn_map: torch.Tensor = query @ key.transpose(-2, -1) * scale_factor
                    if attention_mask is not None:
                        attn_map.masked_fill_(attention_mask.logical_not(), float("-inf"))
                    attn_map = attn_map.softmax(dim=-1)
                    attn_map_collector = self.attention_collector['attn_map'].get(self.name, [])
                    attn_map_collector.append(attn_map.cpu())
                    self.attention_collector['attn_map'][self.name] = attn_map_collector

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if not is_cross_attention:
            if masked_restrict and 0:
                breakpoint()
                attn_map = attn_map.mean(1)
                attn_map = (attn_map == attn_map.max(-1, keepdim=True)[0]).to(attn_map)
                hidden_states = hidden_states * 0. + attn_map @ hidden_states * 1

        if is_cross_attention:
            hidden_states = hidden_states # * 0

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    @staticmethod
    def matmul(query, key, value, heads, attention_mask, return_attn_map=False, attn_threshold=None):
        inner_dim = key.shape[-1]; batch_size = key.shape[0]
        head_dim = inner_dim // heads

        query = query.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, heads, head_dim).transpose(1, 2)
        if value is not None:
            value = value.view(batch_size, -1, heads, head_dim).transpose(1, 2)

        if return_attn_map:
            scale_factor = 1 / math.sqrt(query.size(-1))
            attn_map: torch.Tensor = query @ key.transpose(-2, -1) * scale_factor
            if attention_mask is not None:
                attn_map.masked_fill_(attention_mask.logical_not(), float("-inf"))
            if attn_threshold is not None:
                raise NotImplementedError
                attn_map = (attn_map.detach() > attn_threshold).to(attn_map)
                attn_map[attn_map.sum(-1) == 0] = 1
                attn_map.masked_fill_(attn_map == 0, float("-inf"))

            attn_map_prob = attn_map.softmax(dim=-1)
            if value is not None:
                out = attn_map_prob @ value
            else:
                out = None
        else:
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        if out is not None:
            out = out.transpose(1, 2).reshape(batch_size, -1, heads * head_dim)
            out = out.to(query.dtype)
        if return_attn_map:
            return out, attn_map
        else:
            return out

def set_attn_processor(unet: UNet2DConditionModel, AttnCls, attn_place_in="self_attention",
                       attention_collector: Dict = {}):
    attn_procs = {}
    assert attn_place_in in ["self_attention", "all", "cross_attention"]
    for name, value in unet.attn_processors.items():
        is_cross_attention = not name.endswith("attn1.processor")
        if attn_place_in == "all":
            attn_procs[name] = AttnCls(name=name.split(".transformer_blocks.")[0],
                                       attention_collector=attention_collector)
        elif (attn_place_in == "self_attention") == (not is_cross_attention):
            attn_procs[name] = AttnCls(name=name.split(".transformer_blocks.")[0], attention_collector=attention_collector)
        else:
            attn_procs[name] = value
    unet.set_attn_processor(attn_procs)

if __name__ == "__main__":
    from unhcv.common.utils import human_format_num
    dtype = torch.float16
    query = torch.randn([100, 1000, 1280]).requires_grad_(True).to(dtype).cuda()
    key = torch.randn([100, 1000, 1280]).requires_grad_(True).to(dtype).cuda()
    value = torch.randn([100, 1000, 1280]).requires_grad_(True).to(dtype).cuda()
    print('model_memory 1', human_format_num(torch.cuda.memory_allocated() / 8))

    out = CustomAttnProcessor2_0.matmul(query, key, value, 1, None, return_attn_map=True)
    print('model_memory 2', human_format_num(torch.cuda.memory_allocated() / 8))

    loss = out[1].mean() # + out[0].mean()
    loss.backward()
    print('model_memory 3', human_format_num(torch.cuda.memory_allocated() / 8))

    pass