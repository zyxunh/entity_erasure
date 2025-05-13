import os
from collections import OrderedDict
from dataclasses import dataclass
import os.path as osp
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, Transformer2DModel
from diffusers.configuration_utils import FrozenDict
from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from transformers import CLIPForImageClassification
from transformers.utils import ContextManagers

from unhcv.common.utils import find_path, get_logger
from unhcv.projects.diffusion.ldm.control_attention import ControlAttention2_0
from unhcv.projects.diffusion.ldm.inpainting_attention import set_attn_processor

from unhcv.core.train import AccelerateModelWrap, AccelerateTrain
from unhcv.nn.utils import load_checkpoint, walk_all_children
from unhcv.projects.segmentation.unet_2d_backbone import Unet2DBackbone, LDMInput, BackboneOutput

logger = get_logger(__name__)


def reference_net_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.norm_type == "ada_norm_zero":
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif self.norm_type == "ada_norm_single":
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.squeeze(1)
    else:
        raise ValueError("Incorrect norm used")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    # 1. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    if self.refer_position == "main":
        self.bank.append(norm_hidden_states)
    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
    if self.refer_position == "control":
        control_norm_hidden_states = self.bank[0]
        control_batch_size = control_norm_hidden_states.size(0)
        control_attn_output = self.attn1(
            norm_hidden_states[-control_batch_size:],
            encoder_hidden_states=control_norm_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if attn_output.shape[0] != control_attn_output.shape[0]:
            attn_output[-control_batch_size:] += control_attn_output
        else:
            attn_output = attn_output + control_attn_output
    if self.norm_type == "ada_norm_zero":
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.norm_type == "ada_norm_single":
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 1.2 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

    # 3. Cross-Attention
    if self.attn2 is not None:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    # i2vgen doesn't have this norm 🤷‍♂️
    if self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif not self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm3(hidden_states)

    if self.norm_type == "ada_norm_zero":
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
    else:
        ff_output = self.ff(norm_hidden_states)

    if self.norm_type == "ada_norm_zero":
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.norm_type == "ada_norm_single":
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states


class ReferenceNet(AccelerateModelWrap):
    def __init__(self, model_root, conditioning_channels: int = 5,
                 controlnet_down: bool = True,
                 controlnet_up: bool = True,
                 controlnet_mid: bool = True,
                 fusion_blocks="midup",
                 collector=None):
        super().__init__()
        self.collector = collector
        self.wrap_modules = nn.ModuleDict()
        # with ContextManagers(AccelerateTrain.deepspeed_zero_init_disabled_context_manager()):
        self.wrap_modules["main"] = UNet2DConditionModel.from_pretrained(model_root)
        breakpoint()
        self.wrap_modules["control"] = Unet2DBackbone.from_unet(
                self.wrap_modules["main"],
                conditioning_channels=conditioning_channels,
                controlnet_down=controlnet_down,
                controlnet_up=controlnet_up,
                controlnet_mid=controlnet_mid)
        self.set_reference_net(fusion_blocks=fusion_blocks)

    def set_reference_net(self, fusion_blocks):
        def filter(modules: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
            filter_modules = {}
            for key, value in modules:
                if fusion_blocks == "midup":
                    if key.startswith("mid_block") or key.startswith("up_block"):
                        filter_modules[key] = value
                elif fusion_blocks == "up":
                    if key.startswith("down_blocks"):
                        filter_modules[key] = value
            return filter_modules
        main_transformer_modules = filter(walk_all_children(self.wrap_modules["main"], reserve_cls=BasicTransformerBlock))
        control_transformer_modules = filter(walk_all_children(self.wrap_modules["control"], reserve_cls=BasicTransformerBlock))
        assert len(main_transformer_modules) == len(control_transformer_modules)
        for key in main_transformer_modules.keys():
            main_transformer_modules[key].forward = reference_net_forward
            assert not hasattr(main_transformer_modules[key], "refer_position")
            main_transformer_modules[key].refer_position = "main"
            main_transformer_modules[key].bank = []

            assert not hasattr(main_transformer_modules[key], "collector_name")
            setattr(main_transformer_modules[key], "collector_name", key)

            control_transformer_modules[key].forward = reference_net_forward
            control_transformer_modules[key].refer_position = "control"
            control_transformer_modules[key].bank = main_transformer_modules[key].bank

            assert not hasattr(control_transformer_modules[key], "collector_name")
            setattr(control_transformer_modules[key], "collector_name", key)




    def get_module(self):
        return self.wrap_modules

    def set_module(self, module):
        self.wrap_modules = module


if __name__ == '__main__':
    referenc_net = ReferenceNet(model_root=find_path("model/Realistic_Vision_V4.0_noVAE/unet"))
    breakpoint()
    pass
