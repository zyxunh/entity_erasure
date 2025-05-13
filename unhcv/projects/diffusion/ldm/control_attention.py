import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union, Dict
from collections import defaultdict

import torch
import torch.nn.functional as F
from diffusers.utils import deprecate
from torch import nn
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers import UNet2DConditionModel
# from .attention_utils import CustomAttnProcessor2_0
import kornia
from unhcv.common.image import ratio_length2hw
from unhcv.common.utils import get_logger
from unhcv.nn.transformer import scaled_dot_product_attention
from unhcv.distributed import get_global_rank

logger = get_logger(__name__)


class ControlAttention2_0(nn.Module):
    r"""
    Copy from diffusers v0.29.0.
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name: str, attn: Attention, collector=None, place="main",
                 control_qkv_from_origin=True, outpaint_current=False,
                 original_attn=True, original_attn_unite_entity=False,
                 mask_resize_method="nearest", control_attn=True, share_query=False, train_query=True,
                 key_dim=None, share_to_out=False, control_feature_in_self=False, mean_attn=False,
                 control_feature_name=None):
        super().__init__()
        self.place = place
        self.name = name
        self.control_feature_name = control_feature_name
        self.key_dim = key_dim
        query_dim, inner_dim, cross_attention_dim, bias, out_dim = \
            attn.query_dim, attn.inner_dim, attn.cross_attention_dim, attn.use_bias, attn.out_dim
        self.collector = collector
        self.outpaint_current = outpaint_current
        self.original_attn = original_attn
        self.original_attn_unite_entity = original_attn_unite_entity
        self.mask_resize_method = mask_resize_method
        self.control_attn = control_attn
        self.share_to_out = share_to_out
        self.control_feature_in_self = control_feature_in_self

        if place == "main" and control_attn:
            self.share_query = share_query
            if not share_query:
                self.to_q_control = nn.Linear(query_dim, inner_dim, bias=bias)
            if hasattr(self, "to_q_control"):
                self.to_q_control.requires_grad_(train_query)
            else:
                attn.to_q.requires_grad_(train_query)
            self.to_k_control = nn.Linear(query_dim if key_dim is None else key_dim, inner_dim, bias=bias)
            self.to_v_control = nn.Linear(query_dim if key_dim is None else key_dim, inner_dim, bias=bias)
            if not share_to_out:
                self.to_out_control = nn.Linear(inner_dim, out_dim, bias=True)
            if outpaint_current:
                self.to_outpaint_current = nn.Linear(query_dim if key_dim is None else key_dim, out_dim, bias=not share_to_out)

            assert control_qkv_from_origin
            if control_qkv_from_origin:
                if hasattr(self, "to_q_control"):
                    self.to_q_control.load_state_dict(attn.to_q.state_dict())
                if attn.to_k.in_features != self.to_k_control.in_features:
                    logger.info("attn channels mismatch, not load")
                else:
                    self.to_k_control.load_state_dict(attn.to_k.state_dict())
                    self.to_v_control.load_state_dict(attn.to_v.state_dict())
                if hasattr(self, "to_out_control"):
                    self.to_out_control.load_state_dict(attn.to_out[0].state_dict())
                # self.to_outpaint_current.load_state_dict(attn.to_out[0].state_dict())

        if not original_attn:
            del attn.to_q, attn.to_k, attn.to_out, attn.to_v

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.mean_attn = mean_attn
        if place == "main" and self.mean_attn:
            raise NotImplementedError
            self.to_v_mean = nn.Linear(query_dim if key_dim is None else key_dim, inner_dim, bias=bias)
            self.to_out_mean = nn.Linear(inner_dim, out_dim, bias=True)

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
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        return_attn_mask = self.collector.get("return_attn_mask", False)

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

        if self.original_attn:
            query = attn.to_q(hidden_states)

        if self.place == "control":
            self.collector['_'.join((self.name, "encoder_hidden_states", "control"))] = hidden_states

        if self.place == "main" and self.control_attn:
            if hasattr(self, "to_q_control"):
                query_control = self.to_q_control(hidden_states)
            else:
                query_control = query

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            raise NotImplementedError
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.original_attn:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if hasattr(self, "to_v_control"):
            if self.control_feature_in_self:
                encoder_hidden_states_control = hidden_states
            elif self.control_feature_name is not None:
                encoder_hidden_states_control = self.collector[self.control_feature_name]
            elif self.key_dim is None:
                encoder_hidden_states_control = self.collector['_'.join((self.name, "encoder_hidden_states", "control"))]
            else:
                encoder_hidden_states_control = self.collector["control_feature"]
            key_control = self.to_k_control(encoder_hidden_states_control)
            value_control = self.to_v_control(encoder_hidden_states_control)

        kv_hw_ratio = None
        if self.control_feature_name is not None and self.control_feature_name == 'ip_adapter_embeds':
            kv_hw_ratio = 1

        if self.original_attn:
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if self.original_attn_unite_entity:
                entity_mask = self.collector['entity_mask']
                entity_mask = self.hw2length(entity_mask, query.size(2))
                entity_attention_mask: torch.Tensor = (entity_mask[:, :, :, None] == entity_mask[:, :, None]) & (
                        entity_mask[:, :, :, None] != 0)
                original_attention_mask = entity_attention_mask
                original_attention_mask_valid = original_attention_mask.any(-1)
                original_attention_mask[~original_attention_mask_valid] = True
            else:
                original_attention_mask = None

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = scaled_dot_product_attention(
                query, key, value, attn_mask=original_attention_mask, dropout_p=0.0, is_causal=False, return_attn_mask=return_attn_mask
            )
            if return_attn_mask:
                self.collector['_'.join((f"i_timestep{self.collector['i_timestep']}", self.name, "original_attn_mask"))] = hidden_states[1].cpu()
                hidden_states = hidden_states[0]

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if not self.share_to_out:
                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)
        else:
            hidden_states = None

        if self.place == "main" and self.control_attn:
            inner_dim = key_control.shape[-1]
            head_dim = inner_dim // attn.heads

            batch_size_control = key_control.shape[0]
            if batch_size_control != batch_size:
                query_control = query_control[-batch_size_control:]
            query_control = query_control.view(batch_size_control, -1, attn.heads, head_dim).transpose(1, 2)
            key_control = key_control.view(batch_size_control, -1, attn.heads, head_dim).transpose(1, 2)
            value_control = value_control.view(batch_size_control, -1, attn.heads, head_dim).transpose(1, 2)

            entity_mask = self.collector['entity_mask']
            entity_mask_query = self.hw2length(entity_mask, query_control.size(2))
            entity_mask_key = self.hw2length(entity_mask, key_control.size(2), ratio=kv_hw_ratio)
            entity_attention_mask: torch.Tensor = (entity_mask_query[:, :, :, None] == entity_mask_key[:, :, None]) & (
                        entity_mask_query[:, :, :, None] != 0) & (entity_mask_key[:, :, None] != 0)
            entity_mask_cfg = self.collector.get('entity_mask_cfg', False)
            if entity_mask_cfg:
                raise NotImplementedError
                assert entity_mask_query.size(0) == 2
                entity_attention_mask[0] = ~entity_attention_mask[0]
            inpainting_mask = self.hw2length(self.collector['inpainting_mask'], key_control.size(2), ratio=kv_hw_ratio)
            entity_attention_mask = entity_attention_mask & (inpainting_mask[:, :, None] == 0)
            entity_attention_mask_valid = entity_attention_mask.any(-1)
            entity_attention_mask[~entity_attention_mask_valid] = True

            assert attention_mask is None
            attention_mask_control = entity_attention_mask

            hidden_states_control = scaled_dot_product_attention(
                query_control, key_control, value_control, attn_mask=attention_mask_control, dropout_p=0.0, return_attn_mask=return_attn_mask,
                is_causal=False)
            if return_attn_mask:
                self.collector['_'.join((f"i_timestep{self.collector['i_timestep']}", self.name, "control_attn_mask"))] = hidden_states_control[1].cpu()
                hidden_states_control = hidden_states_control[0]

            hidden_states_control = hidden_states_control * entity_attention_mask_valid[..., None]

            hidden_states_control = hidden_states_control.transpose(1, 2).reshape(batch_size_control, -1, attn.heads * head_dim)
            hidden_states_control = hidden_states_control.to(query_control.dtype)
            if not self.share_to_out:
                hidden_states_control = self.to_out_control(hidden_states_control)

            if self.outpaint_current:
                inpainting_mask = self.collector['inpainting_mask']
                inpainting_mask_hw = inpainting_mask.shape[2:]
                inpainting_mask = self.hw2length(inpainting_mask, query_control.size(2)).transpose(1, 2)
                if encoder_hidden_states_control.size(1) != query_control.size(2):
                    encoder_hidden_states_control_for_current = self.resize_on_length(encoder_hidden_states_control, inpainting_mask_hw, query_control.size(2))
                else:
                    encoder_hidden_states_control_for_current = encoder_hidden_states_control
                hidden_states_control = hidden_states_control * inpainting_mask + (
                            1 - inpainting_mask) * self.to_outpaint_current(encoder_hidden_states_control_for_current)

            if hidden_states is None:
                hidden_states = hidden_states_control
            else:
                if batch_size != batch_size_control:
                    hidden_states[-batch_size_control:] += hidden_states_control
                else:
                    hidden_states = hidden_states + hidden_states_control

        if self.share_to_out:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        if hidden_states is None:
            return 0

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    @staticmethod
    def hw2length(image, length, ratio=None):
        attn_hw = ratio_length2hw(image.shape[2:], ratio=ratio, length=length)
        image: torch.Tensor = F.interpolate(image, attn_hw).flatten(2)
        return image

    @staticmethod
    def resize_on_length(sequence, image_hw, tgt_sequence_length):
        sequence_hw = ratio_length2hw(image_hw, length=sequence.size(1))
        sequence = sequence.view(sequence.size(0), *sequence_hw, sequence.size(2))
        image = sequence.permute(0, 3, 1, 2)
        tgt_sequence_hw = ratio_length2hw(image_hw, length=tgt_sequence_length)
        sequence: torch.Tensor = F.interpolate(image, tgt_sequence_hw).flatten(2).permute(0, 2, 1)
        return sequence

if __name__ == "__main__":
    pass

