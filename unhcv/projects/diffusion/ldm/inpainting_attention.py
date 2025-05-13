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
from .attention_utils import CustomAttnProcessor2_0
import kornia
from unhcv.common.image import ratio_length2hw
from unhcv.distributed import get_global_rank


class InpaintingAttnProcessor2_0(nn.Module, CustomAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, name, attn: Attention,
                 entity_attention=False, entity_attention_config={}, outside_attention=False,
                 outside_attention_config={}, collector=None,
                 original_attention=True, original_attention_config={}, **kwargs):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        query_dim, inner_dim, cross_attention_dim = attn.query_dim, attn.inner_dim, attn.cross_attention_dim
        self.is_cross_attention = attn.is_cross_attention
        self.name = name
        self.collector = collector
        self.entity_attention = entity_attention
        self.outside_attention = outside_attention
        self.original_attention = original_attention

        if entity_attention:
            self.v_entity_attention = entity_attention_config.get("v_entity_attention", True)
            self.attention_map_thres = entity_attention_config.get("attention_map_thres", None)
            self.entity_outside_attention = entity_attention_config.get("outside_attention", False)
            self.entity_attention_smooth = entity_attention_config.get("smooth", None)
            self.entity_attention_smooth_layer = entity_attention_config.get("smooth_layer", None)
            self.entity_share_query = entity_attention_config.get("share_query", False)
            # self.entity_attention_smooth_collector = {}
            assert not self.entity_outside_attention and not self.attention_map_thres, "not implement"
            if not self.entity_share_query:
                self.to_q_entity_attention = nn.Linear(query_dim, inner_dim, attn.use_bias)
            self.to_k_entity_attention = nn.Linear(query_dim, inner_dim, attn.use_bias)
            if self.v_entity_attention:
                self.to_v_entity_attention = nn.Linear(query_dim, inner_dim, attn.use_bias)
            self.entity_attention_heads = 1
        else:
            self.entity_outside_attention = False

        if outside_attention:
            self.outside_attention_fill_learnable_parameter = outside_attention_config.get("fill_learnable_parameter", False)
            self.unite_entity_attention = outside_attention_config.get("unite_entity_attention", False)
            self.to_q_outside_attention = nn.Linear(query_dim, inner_dim, attn.use_bias)
            self.to_k_outside_attention = nn.Linear(query_dim, inner_dim, attn.use_bias)
            self.to_v_outside_attention = nn.Linear(query_dim, inner_dim, attn.use_bias)

            self.to_q_outside_attention.load_state_dict(attn.to_q.state_dict())
            self.to_k_outside_attention.load_state_dict(attn.to_k.state_dict())
            self.to_v_outside_attention.load_state_dict(attn.to_v.state_dict())

            if self.outside_attention_fill_learnable_parameter:
                self.learnable_parameter = nn.Parameter(torch.zeros(inner_dim))
                nn.init.normal(self.learnable_parameter, std=0.2)

        if not original_attention:
            del attn.to_q, attn.to_v, attn.to_k
        self.original_unite_entity_attention = original_attention_config.get("unite_entity_attention", False)

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
        visual_attn = self.collector.get('visual_attn', False) and not torch.is_grad_enabled() and get_global_rank() == 0

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

        assert attention_mask is None
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        attention_mask_original = None

        if 'mask' in self.collector:
            mask = self.collector['mask'].cuda()
            attn_hw = ratio_length2hw(mask.shape[2:], length=sequence_length)
            mask_to_attn_size: torch.Tensor = F.interpolate(mask, attn_hw).flatten(2)
            attention_mask = mask_to_attn_size[:, :, :, None] == mask_to_attn_size[:, :, None]
            attention_mask = (attention_mask).repeat(1, attn.heads, 1, 1)

            inpainting_mask = self.collector["inpainting_mask"].cuda()
            attn_hw = ratio_length2hw(inpainting_mask.shape[2:], length=sequence_length)
            inpainting_mask_to_attn_size: torch.Tensor = F.interpolate(inpainting_mask, attn_hw).flatten(2)
            attention_mask_outside = (inpainting_mask_to_attn_size[:, :, None] == 0).repeat(1, attn.heads,
                                                                                            sequence_length, 1)
            # attention_mask = attention_mask & attention_mask_outside
            # attention_mask[(~attention_mask).all(-1)] = True
            attention_mask_original = attention_mask

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if self.original_attention:
            query = attn.to_q(hidden_states)

        if self.outside_attention:
            query_outside_attention = self.to_q_outside_attention(hidden_states)

        if self.entity_attention:
            if self.entity_share_query:
                query_entity_attention = query
            else:
                query_entity_attention = self.to_q_entity_attention(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states_summary = None
        hidden_states_summary_func = lambda summary, x: x if summary is None else summary + x

        if self.entity_outside_attention or self.outside_attention:
            inpainting_mask = self.collector["inpainting_mask"]
            attn_hw = ratio_length2hw(inpainting_mask.shape[2:], length=sequence_length)
            inpainting_mask_to_attn_size: torch.Tensor = F.interpolate(inpainting_mask, attn_hw).flatten(2)
            attention_mask_outside = (inpainting_mask_to_attn_size[:, :, None] == 0).repeat(1, attn.heads,
                                                                                            sequence_length, 1)
            if attention_mask is not None:
                attention_mask_outside = attention_mask & attention_mask_outside

        if self.entity_attention:
            key_entity_attention = self.to_k_entity_attention(encoder_hidden_states)
            if self.v_entity_attention:
                value_entity_attention = self.to_v_entity_attention(encoder_hidden_states)
            else:
                value_entity_attention = None

            if self.entity_outside_attention:
                attention_mask_entity = attention_mask_outside[:, :self.entity_attention_heads]
            else:
                attention_mask_entity = attention_mask

            hidden_states_entity_attention, entity_attn_map = self.matmul(query=query_entity_attention,
                                                                          key=key_entity_attention,
                                                                          value=value_entity_attention,
                                                                          heads=self.entity_attention_heads,
                                                                          attention_mask=attention_mask_entity,
                                                                          return_attn_map=True,
                                                                          attn_threshold=self.attention_map_thres)
            if visual_attn:
                attn_maps = self.collector.get(self.name + ".entity_attn_map", None)
                if attn_maps is None:
                    attn_maps = []
                    self.collector[self.name + ".entity_attn_map"] = attn_maps
                attn_maps.append(entity_attn_map.cpu())
            else:
                self.collector[self.name + ".entity_attn_map"] = entity_attn_map
            # self.collector["last_entity_attn_map"] = entity_attn_map
            if self.v_entity_attention:
                hidden_states_summary = hidden_states_summary_func(hidden_states_summary, hidden_states_entity_attention)

            if self.entity_attention_smooth is None:
                pass
            else:
                assert False
                i_timestep = self.collector.get("i_timestep", None)
                if i_timestep is not None:
                    if i_timestep == 0:
                        entity_attention_smooth_collector = self.collector.get("attention_smooth", {})
                        if not entity_attention_smooth_collector:
                            self.collector["attention_smooth"] = entity_attention_smooth_collector
                        entity_attention_smooth_collector[self.name] = entity_attn_map
                    else:
                        entity_attention_smooth_collector = self.collector["attention_smooth"]
                        if self.entity_attention_smooth_layer is not None:
                            entity_attn_map_smooth = entity_attention_smooth_collector[self.entity_attention_smooth_layer]
                        else:
                            entity_attn_map_smooth = entity_attention_smooth_collector[self.name]
                        if entity_attn_map.shape != entity_attn_map_smooth.shape:
                            entity_attn_map_hw = ratio_length2hw(inpainting_mask.shape[2:], length=entity_attn_map.size(2))
                            entity_attn_map_smooth_hw = ratio_length2hw(inpainting_mask.shape[2:], length=entity_attn_map_smooth.size(2))
                            entity_attn_map_smooth = self.attn_interpolate(entity_attn_map_smooth, entity_attn_map_smooth_hw, entity_attn_map_hw, mode="nearest")
                        if self.entity_attention_smooth == "first_step":
                            entity_attn_map = entity_attn_map_smooth
                        else:
                            raise NotImplementedError
                attn_maps = self.collector.get(self.name + ".smooth_entity_attn_map", None)
                if attn_maps is None:
                    attn_maps = []
                    self.collector[self.name + ".smooth_entity_attn_map"] = attn_maps
                attn_maps.append(entity_attn_map.cpu())
            entity_attn_map_unite = entity_attn_map.detach() > 0
            entity_attn_map_unite[entity_attn_map_unite.sum(-1) == 0] = True

        if self.original_attention:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            if self.original_unite_entity_attention:
                attention_mask_original = entity_attn_map_unite.repeat(1, attn.heads, 1, 1)

            if "last_entity_attn_map" in self.collector and self.collector["timestep"] == 981 and "down" in self.name:
                assert False
                last_entity_attn_map = self.collector['last_entity_attn_map']
                inpainting_mask = self.collector["inpainting_mask"]
                entity_attn_map_hw = ratio_length2hw(inpainting_mask.shape[2:], length=last_entity_attn_map.size(2))
                attn_map_hw = ratio_length2hw(inpainting_mask.shape[2:], length=sequence_length)
                entity_attn_map_resized = self.attn_interpolate(last_entity_attn_map, entity_attn_map_hw, attn_map_hw)
                attention_mask = (entity_attn_map_resized > 0).repeat(1, attn.heads, 1, 1)
                attention_mask[attention_mask.sum(-1) == 0] = True

            if visual_attn:
                hidden_states, attn_map = self.matmul(query=query, key=key, value=value, heads=attn.heads, attention_mask=attention_mask_original, return_attn_map=True)
                attn_maps = self.collector.get(self.name + ".original_attn_map", [])
                if not attn_maps:
                    self.collector[self.name + ".original_attn_map"] = attn_maps
                attn_maps.append(attn_map.mean(1, keepdim=True).cpu())
            else:
                hidden_states = self.matmul(query=query, key=key, value=value, heads=attn.heads, attention_mask=attention_mask_original)
            hidden_states_summary = hidden_states_summary_func(hidden_states_summary, hidden_states)

        if self.outside_attention:
            key_outside_attention = self.to_k_outside_attention(encoder_hidden_states)
            value_outside_attention = self.to_v_outside_attention(encoder_hidden_states)

            if self.unite_entity_attention:
                attention_mask_outside &= (entity_attn_map.detach() > 0)
            attention_mask_outside[attention_mask_outside.sum(-1) == 0] = True

            if visual_attn:
                hidden_states_outside_attention, attn_map = self.matmul(query=query_outside_attention,
                                                                        key=key_outside_attention,
                                                                        value=value_outside_attention, heads=attn.heads,
                                                                        attention_mask=attention_mask_outside,
                                                                        return_attn_map=True)
                attn_maps = self.collector.get(self.name + ".outside_attn_map", [])
                if not attn_maps:
                    self.collector[self.name + ".outside_attn_map"] = attn_maps
                attn_maps.append(attn_map.cpu())
            else:
                hidden_states_outside_attention: torch.Tensor = self.matmul(query=query_outside_attention, key=key_outside_attention,
                                                              value=value_outside_attention, heads=attn.heads,
                                                              attention_mask=attention_mask_outside)
            if self.outside_attention_fill_learnable_parameter:
                hidden_states_outside_mask = (inpainting_mask_to_attn_size == 0).squeeze(1).unsqueeze(-1)
                hidden_states_outside_attention = hidden_states_outside_attention * (~hidden_states_outside_mask) + \
                                                  self.learnable_parameter[None, None] * hidden_states_outside_mask

            hidden_states_summary = hidden_states_summary_func(hidden_states_summary, hidden_states_outside_attention)

        # linear proj
        hidden_states_summary = attn.to_out[0](hidden_states_summary)
        # dropout
        hidden_states_summary = attn.to_out[1](hidden_states_summary)

        if input_ndim == 4:
            hidden_states_summary = hidden_states_summary.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states_summary = hidden_states_summary + residual

        hidden_states_summary = hidden_states_summary / attn.rescale_output_factor

        return hidden_states_summary

    def maybe_visual_matmul(self, visual_attn, key, **kwargs):
        if visual_attn:
            hidden_states, attn_map = self.matmul(**kwargs, return_attn_map=True)
            attn_maps = self.collector.get(key, [])
            if not attn_maps:
                self.collector[key] = attn_maps
            attn_maps.append(attn_map.cpu())
        else:
            hidden_states = self.matmul(**kwargs)
        return hidden_states

    @staticmethod
    def attn_interpolate(tensor, original_size, size, mode="bilinear"):
        # tensor: N, C, L, L
        # size: (h, w)
        N, C, L, _ = tensor.shape
        L_out = size[0] * size[1]
        tensor = tensor.flatten(1, 2)
        tensor = tensor.view(*tensor.shape[:2], *original_size)
        tensor = F.interpolate(tensor, size, mode=mode)
        tensor = tensor.view(N, C, L, L_out)

        tensor = tensor.transpose(2, 3).flatten(1, 2)
        tensor = tensor.view(*tensor.shape[:2], *original_size)
        tensor = F.interpolate(tensor, size, mode=mode)
        tensor = tensor.view(N, C, L_out, L_out)
        tensor = tensor.transpose(2, 3)
        return tensor


def set_attn_processor(unet: UNet2DConditionModel, attn_cls: Optional[InpaintingAttnProcessor2_0.__class__],
                       attn_place_in="self_attention", attn_cls_config={}, otherwise_attn_cls=None,
                       otherwise_attn_cls_config={}, replace_layers=None,
                       collector: Dict = {}):
    attn_procs = {}
    assert attn_place_in in ["self_attention", "all", "cross_attention"]

    processors = {}
    for name, module in unet.named_modules():
        if hasattr(module, "get_processor"):
            is_cross_attention = module.is_cross_attention
            if not isinstance(module.get_processor(), AttnProcessor2_0):
                breakpoint()
            assert isinstance(module, Attention)
            flag = False
            if replace_layers is not None:
                if isinstance(replace_layers, str):
                    replace_layers = (replace_layers,)
                for var in replace_layers:
                    if var in name:
                        flag = True
                        break
            else:
                flag = True
            if (attn_place_in == "all" or (attn_place_in == "self_attention" and not is_cross_attention) \
                or (attn_place_in == "cross_attention" and is_cross_attention)) \
                    and flag:
                new_processor = attn_cls(name=name, attn=module, **attn_cls_config,
                                                           collector=collector)
                module.set_processor(new_processor)
                processors[f"{name}.processor"] = new_processor
            else:
                if otherwise_attn_cls is not None:
                    raise NotImplementedError
                    processors[f"{name}.processor"] = otherwise_attn_cls(name=name, attn=module,
                                                                         **otherwise_attn_cls_config,
                                                                         collector=collector)
                else:
                    processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)
    # unet.set_attn_processor(processors)


if __name__ == "__main__":
    query = torch.randn([100, 1000, 1280]).requires_grad_(True).to(torch.float).cuda()
    key = torch.randn([100, 1000, 1280]).requires_grad_(True).to(torch.float).cuda()
    value = torch.randn([100, 1000, 1280]).requires_grad_(True).to(torch.float).cuda()
    out = InpaintingAttnProcessor2_0.matmul(query, key, value, 1, return_attn_map=True)
    breakpoint()
    pass

