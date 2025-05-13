# EntityErasure: Erasing Entity Cleanly via Amodal Entity Segmentation and Completion [CVPR2025]

## Introduction
This repository contains the official implementation of the paper EntityErasure.
## Install

Put this project in your ${HOME}/code path

Install mask2former:
```shell
mkdir third_party
git clone https://github.com/zyxunh/Mask2Former_for_entity_erasure.git third_party/Mask2Former
cd third_party/Mask2Former
pip3 install -e .
cd -
```

Install Entity:
```shell
git clone https://github.com/zyxunh/Entity_for_entity_erasure.git third_party/Entity
cd third_party/Entity/Entityv2
bash init.sh
cd -
```

Install diffusers:
```shell
git clone https://github.com/zyxunh/diffusers_for_entity_erasure.git third_party/diffusers
cd third_party/diffusers
pip3 install -e .
cd -
```

Install unhcv:
```shell
pip3 install -e .
```

## Inference

Download model from https://huggingface.co/unhzyx/entity_erasure/tree/main
```shell
export MODEL_ROOT=<downloaded_model_root>

python3 unhcv/projects/diffusion/inpainting/evaluation/evaluation_model.py
```

Metric
download Mask2Former_hornet_3x_576d0b.pth from https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x, then modify third_party/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml WEIGHTS to your model path.
```shell
python3 unhcv/projects/diffusion/inpainting/evaluation/evaluation_metric.py
```

## Finetune
```shell
export MODEL_ROOT=<downloaded_model_root>

# finetune amodal completion model
cd unhcv/projects/diffusion/inpainting
bash train_inpainting.sh 1 --dataset_config configs/dataset/entity_seg.py \
--model_config configs/entity_erasure.py \
--checkpoint ${HOME}/train_outputs/checkpoint/unet_inpainting/release/amodal_completion_model.bin \
--project_suffix finetune

# finetune amodal segmentation model
cd unhcv/projects/segmentation
bash train_mask2former.sh \
--dataset_config ../diffusion/inpainting/configs/dataset/entity_seg.py \
--checkpoint ${HOME}/train_outputs/checkpoint/unet_inpainting/release/amodal_segmentation_model.bin \
--project_suffix finetune
```

## Acknowledgements

This project makes use of the following open-source repositories:

- [diffusers](https://github.com/huggingface/diffusers)
- [Entity Segmentation](https://github.com/qqlu/Entity)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)

We sincerely thank the authors for open-sourcing their valuable work.


## Reference

If you use this codebase or models in your research, please consider cite .

```
@inproceedings{zhu2025entityerasure,
  title={EntityErasure: Erasing Entity Cleanly via Amodal Entity Segmentation and Completion},
  author={Zhu, Yixing and Zhang, Qing and Wang, Yitong and Nie, Yongwei and Zheng, Wei-Shi},
  booktitle={CVPR},
  year={2025}
}

```