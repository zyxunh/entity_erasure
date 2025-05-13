# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from diffusers 0.29.2

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
    get_mid_block,
    get_up_block
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from unhcv.nn.utils import load_checkpoint
from unhcv.nn.utils.analyse import cal_para_num

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class LDMInput:
    sample: torch.Tensor
    timestep: torch.Tensor
    encoder_hidden_states: torch.Tensor = None
    controlnet_cond: torch.Tensor = None
    output_hw: List = None

    @property
    def shape(self):
        return self.sample.shape

    @property
    def device(self):
        return self.sample.device


@dataclass
class BackboneOutput(BaseOutput):
    """
    The output of [`Unet2DBackbone`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor] = None
    mid_block_res_samples: torch.Tensor = None
    feature_maps: List[torch.Tensor] = None
    controlnet_up_block_res_samples: Tuple[torch.Tensor] = None
    controlnet_down_block_res_samples: Tuple[torch.Tensor] = None
    controlnet_mid_block_res_sample: torch.Tensor = None


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class Unet2DBackbone(UNet2DConditionModel, ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    A ControlNet model.

    Args:
        in_channels (`int`, defaults to 4):
            The number of channels in the input sample.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, defaults to 0):
            The frequency shift to apply to the time embedding.
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        downsample_padding (`int`, defaults to 1):
            The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, defaults to 1):
            The scale factor to use for the mid block.
        act_fn (`str`, defaults to "silu"):
            The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the normalization. If None, normalization and activation layers is skipped
            in post-processing.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon to use for the normalization.
        cross_attention_dim (`int`, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`Union[int, Tuple[int]]`, defaults to 8):
            The dimension of the attention heads.
        use_linear_projection (`bool`, defaults to `False`):
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from None,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to 0):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        upcast_attention (`bool`, defaults to `False`):
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            Time scale shift config for ResNet blocks (see `ResnetBlock2D`). Choose from `default` or `scale_shift`.
        projection_class_embeddings_input_dim (`int`, *optional*, defaults to `None`):
            The dimension of the `class_labels` input when `class_embed_type="projection"`. Required when
            `class_embed_type="projection"`.
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
        global_pool_conditions (`bool`, defaults to `False`):
            TODO(Patrick) - unused parameter.
        addition_embed_type_num_heads (`int`, defaults to 64):
            The number of heads to use for the `TextTimeEmbedding` layer.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conv_in_on: bool = True,
        out_channels: int = 4,
        center_input_sample: bool = False,
        conditioning_channels: int = 5,
        condition_embed_type: str = "conv",
        return_down_block_indexes: Tuple[int, ...] = (),
        return_up_block_indexes: Tuple[int, ...] = (14, 10, 6, 2), # (2, 6, 10, 14),
        return_mid_block: bool = False,
        controlnet_down: bool = False,
        controlnet_up: bool = False,
        controlnet_mid: bool = False,
        global_pool_conditions: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads: int = 64,
    ):
        ModelMixin.__init__(self)

        # self.sample_size = sample_size

        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        self._check_config(
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            reverse_transformer_layers_per_block=reverse_transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
        )

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        if conv_in_on:
            self.conv_in = nn.Conv2d(
                in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
            )
        if condition_embed_type == "conv":
            self.conv_condition_in = nn.Conv2d(
                conditioning_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
            )
        elif condition_embed_type == "controlnet_embed":
            self.conv_condition_in = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=block_out_channels[0],
                conditioning_channels=conditioning_channels)
        elif condition_embed_type == 'none':
            self.conv_condition_in = nn.Identity()

        # time
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            time_embedding_type,
            block_out_channels=block_out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embedding_dim=time_embedding_dim,
        )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        self._set_encoder_hid_proj(
            encoder_hid_dim_type,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
        )

        # class embedding
        self._set_class_embedding(
            class_embed_type,
            act_fn=act_fn,
            num_class_embeds=num_class_embeds,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
            timestep_input_dim=timestep_input_dim,
        )

        self._set_add_embedding(
            addition_embed_type,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
            addition_time_embed_dim=addition_time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            encoder_hid_dim=encoder_hid_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            time_embed_dim=time_embed_dim,
        )

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if controlnet_down:
            self.controlnet_down_blocks = nn.ModuleList([])
        if controlnet_up:
            self.controlnet_up_blocks = nn.ModuleList([])
        self.controlnet_down = controlnet_down
        self.controlnet_up = controlnet_up
        self.controlnet_mid = controlnet_mid

        self._channels = {}

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if not isinstance(cross_attention_dim, (tuple, list)):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]

        down_block_index = 0
        if down_block_index in self.return_down_block_indexes:
            raise NotImplementedError
            controlnet_block = self.build_controlnet_block(output_channel, output_channel, kernel_size=1)
            self.controlnet_down_blocks.append(controlnet_block)
            self._channels.append(output_channel)
        down_block_index += 1

        if controlnet_down:
            self.controlnet_down_blocks.append(self.build_controlnet_block(output_channel, output_channel))

        self.return_down_block_indexes = return_down_block_indexes
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

            outs_per_block = layers_per_block[i]
            if not is_final_block:
                outs_per_block += 1
            for i_res_down_block in return_down_block_indexes:
                if down_block_index <= i_res_down_block < down_block_index + outs_per_block:
                    # if not is_final_block:
                    # controlnet_block = self.build_controlnet_block(output_channel, output_channel, kernel_size=1)
                    # self.controlnet_down_blocks.append(controlnet_block)
                    self._channels.append(output_channel)
            down_block_index += outs_per_block

            if controlnet_down:
                for _ in range(layers_per_block[i]):
                    self.controlnet_down_blocks.append(self.build_controlnet_block(output_channel, output_channel))

                if not is_final_block:
                    self.controlnet_down_blocks.append(self.build_controlnet_block(output_channel, output_channel))

        # if not is_final_block:
        #     controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        #     controlnet_block = zero_module(controlnet_block)
        #     self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = block_out_channels[-1]

        if controlnet_mid:
            self.controlnet_mid_block = self.build_controlnet_block(output_channel, output_channel)

        if return_mid_block:
            controlnet_block = self.build_controlnet_block(mid_block_channel, mid_block_channel, kernel_size=1)
            self.controlnet_mid_block = controlnet_block
            self._channels.append(mid_block_channel)
        self.return_mid_block = return_mid_block


        self.mid_block = get_mid_block(
            mid_block_type,
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            output_scale_factor=mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            resnet_skip_time_act=resnet_skip_time_act,
            cross_attention_norm=cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],
            dropout=dropout,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]

        self.return_up_block_indexes = return_up_block_indexes
        up_block_index = 0
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

            if controlnet_up:
                for _ in range(layers_per_block[i] + 1):
                    self.controlnet_up_blocks.append(self.build_controlnet_block(output_channel, output_channel))

                if not is_final_block:
                    self.controlnet_up_blocks.append(self.build_controlnet_block(output_channel, output_channel))

            outs_per_block = reversed_layers_per_block[i] + 1
            if not is_final_block:
                outs_per_block += 1
            for i_res_up_block in return_up_block_indexes:
                if up_block_index <= i_res_up_block < up_block_index + outs_per_block:
                    # if not is_final_block:
                    # controlnet_block = self.build_controlnet_block(output_channel, output_channel, kernel_size=1)
                    # self.controlnet_down_blocks.append(controlnet_block)
                    self._channels[i_res_up_block] = output_channel
            up_block_index += outs_per_block

        # out
        # if norm_num_groups is not None:
        #     self.conv_norm_out = nn.GroupNorm(
        #         num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
        #     )
        #
        #     self.conv_act = get_activation(act_fn)
        #
        # else:
        #     self.conv_norm_out = None
        #     self.conv_act = None
        #
        # conv_out_padding = (conv_out_kernel - 1) // 2
        # self.conv_out = nn.Conv2d(
        #     block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        # )
        #
        # self._set_pos_net_if_use_gligen(attention_type=attention_type, cross_attention_dim=cross_attention_dim)

    @classmethod
    def from_unet(
            cls,
            unet: UNet2DConditionModel,
            load_weights_from_unet: bool = True,
            conditioning_channels: int = 3,
            conv_condition_in_from_unet: bool = True,
            missing_resolve_function=None,
            retain_cross_attention=False,
            **kwargs
    ):
        r"""
        Instantiate a [`Unet2DBackbone`] from [`UNet2DConditionModel`].

        Parameters:
            unet (`UNet2DConditionModel`):
                The UNet model weights to copy to the [`Unet2DBackbone`]. All configuration options are also copied
                where applicable.
        """
        transformer_layers_per_block = (
            unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
        )
        encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
        encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
        addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
        addition_time_embed_dim = (
            unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
        )

        controlnet = cls(
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=unet.config.in_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            only_cross_attention=unet.config.only_cross_attention,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim if retain_cross_attention else None,
            attention_head_dim=unet.config.attention_head_dim,
            num_attention_heads=unet.config.num_attention_heads,
            use_linear_projection=unet.config.use_linear_projection,
            class_embed_type=unet.config.class_embed_type,
            num_class_embeds=unet.config.num_class_embeds,
            upcast_attention=unet.config.upcast_attention,
            resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
            mid_block_type=unet.config.mid_block_type,
            conditioning_channels=conditioning_channels,
            **kwargs
        )

        if load_weights_from_unet:
            # print(controlnet.load_state_dict(unet.state_dict(), strict=False))
            random_init = True

            def missing_resolve_function_default(key, parameter, state_dict):
                if 'conv_condition_in' in key:
                    state: torch.Tensor = state_dict[key.replace('conv_condition_in', 'conv_in')]
                    if state.dim() == 4:
                        parameter[:, :state.size(1)] = state
                    elif state.dim() == 1:
                        parameter = state
                    else:
                        raise NotImplementedError
                    print(f"{key} load with missing resolve function")
                return parameter
            if missing_resolve_function is None:
                missing_resolve_function = missing_resolve_function_default

            load_checkpoint(controlnet, unet.state_dict(), log_missing_keys=True, missing_resolve_function=missing_resolve_function)
            # controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            # controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            # controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())
            #
            # if controlnet.class_embedding:
            #     controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())
            #
            # if hasattr(controlnet, "add_embedding"):
            #     controlnet.add_embedding.load_state_dict(unet.add_embedding.state_dict())
            #
            # print(controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict(), strict=False))
            # print(controlnet.mid_block.load_state_dict(unet.mid_block.state_dict(), strict=False))

        return controlnet

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int] = None,
        encoder_hidden_states: torch.Tensor = None,
        controlnet_cond: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[BackboneOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        """
        The [`Unet2DBackbone`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnet.BackboneOutput`] instead of a plain tuple.

        Returns:
            [`~models.controlnet.BackboneOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnet.BackboneOutput`] is returned, otherwise a tuple is
                returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None
        if isinstance(sample, LDMInput):
            sample, timestep, controlnet_cond, encoder_hidden_states = sample.sample, sample.timestep, sample.controlnet_cond, sample.encoder_hidden_states

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        if hasattr(self, "conv_in"):
            sample = self.conv_in(sample)
        else:
            sample = None

        condition_input = self.conv_condition_in(controlnet_cond)
        sample = condition_input if sample is None else sample + condition_input

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if self.controlnet_down:
            controlnet_down_block_res_samples = ()
            for block, res_sample in zip(self.controlnet_down_blocks, down_block_res_samples):
                controlnet_down_block_res_samples += (block(res_sample),)
        else:
            controlnet_down_block_res_samples = None

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        if self.controlnet_mid:
            controlnet_mid_block_res_sample = self.controlnet_mid_block(sample)
        else:
            controlnet_mid_block_res_sample = None

        # 5. Control net blocks

        # controlnet_down_block_res_samples = ()

        # return_down_block_indexes: 2, 3, 5, 8
        # controlnet_down_block_res_samples = [down_block_res_samples[var] for var in self.return_down_block_indexes]
        # down_block_last_samples_part = [down_block_res_samples[var] for var in self.return_down_block_indexes]
        # for down_block_res_sample, controlnet_block in zip(down_block_last_samples_part, self.controlnet_down_blocks):
        #     down_block_res_sample = controlnet_block(down_block_res_sample)
        #     controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        mid_block_res_samples = ()
        if self.return_mid_block:
            mid_block_res_sample = self.controlnet_mid_block(sample)
            mid_block_res_samples += (mid_block_res_sample)

        # 5. up
        up_block_res_samples = ()
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample, up_res_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_res_samples=True,
                    cat_res_samples=False
                )
            else:
                sample, up_res_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    return_res_samples=True,
                    cat_res_samples=False
                )

            up_block_res_samples += up_res_samples

        # 2, 6, 10, 14
        feature_maps = [up_block_res_samples[var] for var in self.return_up_block_indexes]

        if self.controlnet_up:
            controlnet_up_block_res_samples = ()
            for block, res_sample in zip(self.controlnet_up_blocks, up_block_res_samples):
                controlnet_up_block_res_samples += (block(res_sample),)
        else:
            controlnet_up_block_res_samples = None
        # 6. post-process
        # if self.conv_norm_out:
        #     sample = self.conv_norm_out(sample)
        #     sample = self.conv_act(sample)
        # sample = self.conv_out(sample)
        #
        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)
        #
        # if not return_dict:
        #     return (sample,)
        #
        # return UNet2DConditionOutput(sample=sample)

        # 6. scaling
        # if guess_mode and not self.config.global_pool_conditions:
        #     scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
        #     scales = scales * conditioning_scale
        #     down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
        #     mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        # else:
        #     down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        #     mid_block_res_sample = mid_block_res_sample * conditioning_scale
        #
        # if self.config.global_pool_conditions:
        #     down_block_res_samples = [
        #         torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
        #     ]
        #     mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(controlnet_down_block_res_samples) + 1 + len(controlnet_up_block_res_samples), device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale

            if controlnet_down_block_res_samples is not None:
                controlnet_down_block_res_samples = [sample * scale for sample, scale in zip(controlnet_down_block_res_samples, scales[:len(brushnet_down_block_res_samples)])]
                controlnet_mid_block_res_sample = controlnet_mid_block_res_sample * scales[len(controlnet_down_block_res_samples)]
                controlnet_up_block_res_samples = [sample * scale for sample, scale in zip(controlnet_up_block_res_samples, scales[len(controlnet_down_block_res_samples)+1:])]
        else:
            if controlnet_down_block_res_samples is not None:
                controlnet_down_block_res_samples = [sample * conditioning_scale for sample in controlnet_down_block_res_samples]
                controlnet_mid_block_res_sample = controlnet_mid_block_res_sample * conditioning_scale
                controlnet_up_block_res_samples = [sample * conditioning_scale for sample in controlnet_up_block_res_samples]


        if not return_dict:
            return feature_maps
            # return (down_block_res_samples, mid_block_res_sample)

        return BackboneOutput(feature_maps=feature_maps,
                              controlnet_down_block_res_samples=controlnet_down_block_res_samples,
                              controlnet_up_block_res_samples=controlnet_up_block_res_samples,
                              controlnet_mid_block_res_sample=controlnet_mid_block_res_sample)

    @property
    def names(self):
        _names = []
        for i in range(len(self.channels)):
            _names.append(f"res{i + 2}")
        return _names

    @property
    def channels(self):
        channels = [self._channels[var] for var in self.return_up_block_indexes]
        return channels

    @property
    def strides(self):
        stride = 4
        _strides = []
        for i in range(len(self.channels)):
            _strides.append(stride)
            stride *= 2
        return _strides

    def build_controlnet_block(self, in_channel, out_channel, kernel_size=1):
        controlnet_block = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size)
        controlnet_block = zero_module(controlnet_block)
        return controlnet_block

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x)


class ResnetBlock2D(ResnetBlock2D):
    def __init__(self, in_channels, out_channels, temb_channels=None, non_linearity="silu", conv_shortcut=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, temb_channels=temb_channels, non_linearity=non_linearity, conv_shortcut=conv_shortcut)

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        return super().forward(input_tensor, temb=temb)

@dataclass
class IndexConfig:
    indexes: [Tuple[int], int]
    out_channels: int = None
    scales: Optional[Union[Tuple[int], int]] = None
    module_cls: type = ResnetBlock2D
    module_cls_config: Dict = None

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
        # if isinstance(self.indexes, int):
        #     self.indexes = [self.indexes]
        # if isinstance(self.scales, int):
        #     self.scales = [self.scales]


class FPNBackbone(nn.Module):
    def __init__(self, backbone: Unet2DBackbone, fpn_indexes_config: List[IndexConfig]) -> None:
        super().__init__()
        self.backbone = backbone
        self.fpn_modules = nn.ModuleList()
        self.fpn_indexes = []
        self.num_modules = len(fpn_indexes_config)
        self._channels = []
        for fpn_index_config in fpn_indexes_config:
            modules = nn.ModuleList()
            self.fpn_modules.append(modules)
            out_channels = fpn_index_config.out_channels
            self.fpn_indexes.append(fpn_index_config.indexes)
            for i_index in range(len(fpn_index_config.indexes)):
                scale = fpn_index_config.scales[i_index] if fpn_index_config.scales else 1
                index = fpn_index_config.indexes[i_index]
                in_channels = backbone.channels[index]
                if out_channels is None:
                    out_channels = in_channels
                module_cls_config = {} if fpn_index_config.module_cls_config is None else fpn_index_config.module_cls_config
                module = []
                if scale != 1:
                    module.append(Upsample2D(channels=in_channels, use_conv=True))
                    # module.append(nn.Upsample(scale_factor=scale))
                module.append(fpn_index_config.module_cls(in_channels=in_channels, out_channels=out_channels, **module_cls_config))
                modules.append(nn.Sequential(*module))
            self._channels.append(out_channels)

    def feature_extract(self, model_input):
        features = self.backbone(model_input, return_dict=False)
        return features

    def forward(self, model_input):
        features = self.feature_extract(model_input)
        new_features = ()
        for modules, indexes in zip(self.fpn_modules, self.fpn_indexes):
            feature_res = None
            for module, index in zip(modules, indexes):
                feature = module(features[index])
                if feature_res is None:
                    feature_res = feature
                else:
                    feature_res = feature_res + feature
            new_features += (feature_res,)
        return new_features

    @property
    def names(self):
        _names = []
        for i in range(self.num_modules):
            _names.append(f"res{i + 2}")
        return _names

    @property
    def strides(self):
        stride = 4
        _strides = []
        for i in range(self.num_modules):
            _strides.append(stride)
            stride *= 2
        return _strides

    @property
    def channels(self):
        return self._channels

if __name__ == '__main__':
    # k = LDMInput(pixel_values=1, timesteps=2)
    control = ControlNetConditioningEmbedding(conditioning_embedding_channels=320)
    print(cal_para_num(control))
    print(cal_para_num(nn.Conv2d(30, 320, kernel_size=8, stride=8)))
    breakpoint()
    backbone = Unet2DBackbone.from_unet(
        UNet2DConditionModel.from_pretrained("/home/tiger/model/realisticVisionV60B1_v51VAE/unet"),
        conditioning_channels=5,
        controlnet_down=True,
        controlnet_up=False,
        controlnet_mid=True,
        missing_resolve_function=lambda *args, **kwargs: None)

    sample = torch.zeros([1, 4, 64, 64], dtype=torch.float)
    condtion = torch.zeros([1, 5, 64, 64], dtype=torch.float)
    model_input = LDMInput(sample=sample, controlnet_cond=condtion, encoder_hidden_states=None, timestep=1)
    out = backbone(model_input)
    breakpoint()


    fpn_indexes_config = [IndexConfig(indexes=(0,), scales=(2,), out_channels=512),
                          IndexConfig(indexes=(0,), out_channels=512), IndexConfig(indexes=(1,), out_channels=512),
                          IndexConfig(indexes=(2, 3), scales=(1, 2), out_channels=512)]
    fpn_backbone = FPNBackbone(backbone, fpn_indexes_config=fpn_indexes_config)

    sample = torch.zeros([1, 4, 64, 64], dtype=torch.float)
    condtion = torch.zeros([1, 5, 64, 64], dtype=torch.float)
    model_input = LDMInput(sample=sample, controlnet_cond=condtion, encoder_hidden_states=None, timestep=1)
    out = fpn_backbone(model_input)
    FPNBackbone
    pass
