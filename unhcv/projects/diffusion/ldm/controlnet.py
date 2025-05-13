import os
from collections import OrderedDict
from dataclasses import dataclass
import os.path as osp

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, Transformer2DModel
from diffusers.configuration_utils import FrozenDict
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from transformers.utils import ContextManagers

from .transformer_2d_backbone import PixArtTransformer2DModelBackbone
from unhcv.projects.diffusion.ldm.control_attention import ControlAttention2_0
from unhcv.projects.diffusion.ldm.inpainting_attention import set_attn_processor

from unhcv.core.train import AccelerateModelWrap, AccelerateTrain
from unhcv.nn.utils import load_checkpoint
from unhcv.projects.segmentation.unet_2d_backbone import Unet2DBackbone, LDMInput, BackboneOutput


HOME = os.environ["HOME"]


@dataclass
class ControlNetConfig:
    models_root: str = '/home/tiger/model/realisticVisionV60B1_v51VAE'
    conditioning_channels: int = 5
    controlnet_down: bool = True
    controlnet_up: bool = True
    controlnet_mid: bool = True
    missing_resolve_function: type = lambda *args, **kwargs: None


class ControlNet(AccelerateModelWrap):
    control_branch_on = True

    def __init__(self, models_root: str = '/home/tiger/model/realisticVisionV60B1_v51VAE',
                 conditioning_channels: int = 5,
                 controlnet_down: bool = True,
                 controlnet_up: bool = True,
                 controlnet_mid: bool = True,
                 missing_resolve_function: type = lambda *args, **kwargs: None,
                 control_branch_conv_in_on=True,
                 retain_cross_attention=False,
                 control_branch_config={}, condition_embed_type='none', dtype=None, collector=None,
                 outpaint_current=True):
        super().__init__()
        with ContextManagers(AccelerateTrain.deepspeed_zero_init_disabled_context_manager()):
            self.main_branch = UNet2DConditionModel.from_pretrained(osp.join(models_root, "unet"))
        if self.control_branch_on:
            self.control_branch: Unet2DBackbone = Unet2DBackbone.from_unet(
                self.main_branch,
                conditioning_channels=conditioning_channels,
                controlnet_down=controlnet_down,
                controlnet_up=controlnet_up,
                controlnet_mid=controlnet_mid,
                missing_resolve_function=missing_resolve_function,
                condition_embed_type=condition_embed_type,
                conv_in_on=control_branch_conv_in_on,
                retain_cross_attention=retain_cross_attention,
                **control_branch_config)

        self.dtype = dtype
        self.main_branch.requires_grad_(False)

    def __call__(self, sample, timestep, encoder_hidden_states, controlnet_cond=None, attention_mask=None):
        controlnet_cond = self.extract_control_feature(controlnet_cond)
        control_features = self.control_branch(sample=sample[:, :4], timestep=timestep, controlnet_cond=controlnet_cond,
                                               attention_mask=attention_mask, conditioning_scale=0.5)
        down_block_add_samples = [var.to(self.dtype) for var in control_features.controlnet_down_block_res_samples] \
                                      if control_features.controlnet_down_block_res_samples is not None else None
        mid_block_add_sample = control_features.controlnet_mid_block_res_sample.to(self.dtype)
        up_block_add_samples = [var.to(self.dtype) for var in control_features.controlnet_up_block_res_samples] \
                                       if control_features.controlnet_up_block_res_samples is not None else None
        output = self.main_branch(sample=sample, timestep=timestep,
                                  encoder_hidden_states=encoder_hidden_states,
                                  down_block_add_samples=down_block_add_samples,
                                  mid_block_add_sample=mid_block_add_sample,
                                  up_block_add_samples=up_block_add_samples,
                                  attention_mask=attention_mask)
        return output

    def extract_control_feature(self, controlnet_cond):
        return controlnet_cond

    @property
    def trained_models(self):
        return self.control_branch

    @property
    def frozen_models(self):
        return [self.main_branch]


@dataclass
class EntityInpaintingCond:
    entity: torch.Tensor
    outpainting_image: torch.Tensor


class EntityInpaitingControlNet(ControlNet):
    def __init__(self, *args, entity_in_channels=None, cond_embedding_dim=320,
                 cond_stride=8, entity_control=True, outpainting_image_control=True,
                 entity_conv_in_type="default", **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_control = entity_control
        self.outpainting_image_control = outpainting_image_control
        self.entity_in_channels = entity_in_channels
        if entity_control:
            if entity_conv_in_type == "default":
                self.control_branch.entity_conv_in = nn.Conv2d(entity_in_channels, cond_embedding_dim,
                                                               kernel_size=cond_stride,
                                                               stride=cond_stride)
            elif entity_conv_in_type == "stack_convs":
                self.control_branch.entity_conv_in = ControlNetConditioningEmbedding(
                    conditioning_embedding_channels=cond_embedding_dim, conditioning_channels=entity_in_channels)

        if outpainting_image_control:
            self.control_branch.outpainting_image_conv_in = nn.Conv2d(5, cond_embedding_dim,
                                                       kernel_size=self.main_branch.config.conv_in_kernel,
                                                       padding=self.main_branch.config.conv_in_kernel // 2)

        if outpainting_image_control:
            load_checkpoint(self.control_branch.outpainting_image_conv_in, self.main_branch.conv_in.state_dict(), log_missing_keys=True,
                            mismatch_resolve_function=self.mismatch_resolve_function, mismatch_shape=True)

    @staticmethod
    def mismatch_resolve_function(key, state_parameter, model_parameter: torch.Tensor):
            if model_parameter.dim() == 4:
                model_parameter[:, :4] = state_parameter
            else:
                model_parameter[:4] = state_parameter
            return model_parameter

    def extract_control_feature(self, controlnet_cond: EntityInpaintingCond):
        feature = None
        if self.entity_control:
            feature = self.control_branch.entity_conv_in(controlnet_cond.entity)
        if self.outpainting_image_control:
            feature_res = self.control_branch.outpainting_image_conv_in(controlnet_cond.outpainting_image)
            if feature is None:
                feature = feature_res
            else:
                feature = feature_res + feature
        return feature


class AttentionControlNet(nn.Module, ControlNet):

    def __init__(self, models_root: str = '/home/zhuyixing/model/PowerPaint-v2-1/realisticVisionV60B1_v51VAE',
                 conditioning_channels: int = 5, controlnet_down: bool = True, controlnet_up: bool = True,
                 controlnet_mid: bool = True, missing_resolve_function: type = lambda *args, **kwargs: None,
                 collector={}, control_branch_conv_in_on=False, outpaint_current=True,
                 original_attn=True, original_attn_unite_entity=False,
                 main_branch_requires_grad=False, inject_place="self_attention", share_query=False, train_query=True,
                 control_branch_config={}, control_feature_dynamic=True, share_to_out=False,
                 control_branch_on=True, ip_adapter_on=False, key_dim=None, attn_config={},
                 entity_rgb_cond_on=False, control_timestep=None):
        super().__init__()
        self.collector = collector
        self.control_feature_dynamic = control_feature_dynamic
        if not control_feature_dynamic:
            self.control_feature_head = nn.LayerNorm(1280)
        retain_cross_attention = inject_place == "cross_attention"
        self.control_branch_on = control_branch_on
        ControlNet.__init__(self, models_root, conditioning_channels, controlnet_down, controlnet_up, controlnet_mid,
                            missing_resolve_function, control_branch_conv_in_on=control_branch_conv_in_on, retain_cross_attention=retain_cross_attention,
                            control_branch_config=control_branch_config)
        if key_dim is None and not control_feature_dynamic:
            key_dim = 1280
        if inject_place is not None:
            set_attn_processor(self.main_branch, attn_cls=ControlAttention2_0,
                               attn_cls_config=dict(place='main', outpaint_current=outpaint_current,
                                                    original_attn=original_attn,
                                                    original_attn_unite_entity=original_attn_unite_entity,
                                                    share_query=share_query, train_query=train_query, key_dim=key_dim,
                                                    share_to_out=share_to_out, **attn_config),
                               collector=collector, attn_place_in=inject_place)
        if self.control_branch_on:
            if inject_place is not None:
                set_attn_processor(self.control_branch, attn_cls=ControlAttention2_0,
                                   attn_cls_config=dict(place='control', original_attn=inject_place == "self_attention"),
                                   collector=collector, attn_place_in=inject_place)
            self.control_branch.control_conv_in = nn.Conv2d(conditioning_channels,
                                                            self.main_branch.config.block_out_channels[0],
                                                            kernel_size=self.main_branch.config.conv_in_kernel,
                                                            padding=self.main_branch.config.conv_in_kernel // 2)
            load_checkpoint(self.control_branch.control_conv_in, self.main_branch.conv_in.state_dict(),
                            log_missing_keys=True,
                            mismatch_resolve_function=EntityInpaitingControlNet.mismatch_resolve_function,
                            mismatch_shape=True)

            # rgb for ablation
            self.entity_rgb_cond_on = entity_rgb_cond_on
            if entity_rgb_cond_on:
                self.control_branch.rgb_control_in = ControlNetConditioningEmbedding(
                    conditioning_embedding_channels=self.main_branch.config.block_out_channels[0],
                    conditioning_channels=3)

        self.ip_adapter_on = ip_adapter_on
        if ip_adapter_on:
            from ip_adapter import CustomIPAdapter
            self.ip_adapter = CustomIPAdapter(
                unet_config=self.main_branch.config,
                image_encoder_path=os.path.join(HOME, "model/IP-Adapter/CLIP-ViT-H-14-laion2B-s32B-b79K"),
                ip_ckpt=None)

        self.control_timestep = control_timestep

        if main_branch_requires_grad:
            self.main_branch.requires_grad_(True)

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond=None, attention_mask=None,
                ip_adapter_image=None):
        if self.control_branch_on:
            controlnet_cond = self.extract_control_feature(controlnet_cond)
            sample_control = sample
            if timestep.shape[0] != controlnet_cond.shape[0]:
                timestep = timestep[-controlnet_cond.shape[0]:]
                sample_control = sample_control[-controlnet_cond.shape[0]:]
            control_timestep = timestep
            if self.control_timestep is not None:
                control_timestep = control_timestep.new_tensor(self.control_timestep).repeat(len(control_timestep))
            control_features = self.control_branch(sample=sample_control[:, :4], timestep=control_timestep,
                                                   controlnet_cond=controlnet_cond,
                                                   attention_mask=attention_mask, conditioning_scale=1)
        else:
            control_features = BackboneOutput()
        if self.ip_adapter_on:
            self.collector['ip_adapter_embeds'] = self.ip_adapter.get_image_embeds(ip_adapter_image)

        if not self.control_feature_dynamic:
            control_feature = control_features.feature_maps[0].flatten(2).permute(0, 2, 1)
            control_feature = self.control_feature_head(control_feature)
            self.collector['control_feature'] = control_feature
        output = self.main_branch(sample=sample, timestep=timestep,
                                  encoder_hidden_states=encoder_hidden_states,
                                  down_block_add_samples=list(
                                      control_features.controlnet_down_block_res_samples) if control_features.controlnet_down_block_res_samples is not None else None,
                                  mid_block_add_sample=control_features.controlnet_mid_block_res_sample,
                                  up_block_add_samples=list(
                                      control_features.controlnet_up_block_res_samples) if control_features.controlnet_up_block_res_samples is not None else None,
                                  attention_mask=attention_mask)
        return output

    def extract_control_feature(self, controlnet_cond):
        if self.entity_rgb_cond_on:
            breakpoint()
            feature = self.control_branch.rgb_control_in(controlnet_cond[0]) #+ \
                      # self.control_branch.control_conv_in(controlnet_cond[1])
        else:
            feature = self.control_branch.control_conv_in(controlnet_cond)
        return feature


if __name__ == '__main__':
    sample = torch.zeros([1, 4, 64, 64], dtype=torch.float)
    condition = torch.zeros([1, 5, 64, 64], dtype=torch.float)
    entity_condition = torch.zeros([1, 30, 512, 512], dtype=torch.float)
    encoder_hidden_states = torch.zeros(1, 77, 768, dtype=torch.float)
    collector = {}
    attention_control = AttentionControlNet(conditioning_channels=4, controlnet_up=False, controlnet_down=False,
                                            controlnet_mid=False, collector=collector)
    collector['entity_mask'] = torch.zeros([1, 1, 512, 512], dtype=torch.float)
    collector['inpainting_mask'] = torch.zeros([1, 1, 512, 512], dtype=torch.float)
    out = attention_control(sample, 1, encoder_hidden_states, sample)
    breakpoint()
    entity_inpainting_controlnet = EntityInpaitingControlNet(entity_in_channels=30)

    out = entity_inpainting_controlnet(sample, timestep=1, encoder_hidden_states=encoder_hidden_states,
                                       controlnet_cond=EntityInpaintingCond(entity=entity_condition, outpainting_image=condition))
    breakpoint()