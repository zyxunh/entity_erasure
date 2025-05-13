import argparse

import numpy as np
import torch
from PIL import Image
from CropFormer.api import EntityApi
from torchvision.transforms.functional import pil_to_tensor

from unhcv.common.types import DataDict
from unhcv.common.utils import find_path, obj_load


class RemovingMetric:
    def __init__(self, instance_score_thres=0.3, inter_ratio_thres=0.8, min_area_ratio_thres=0.005):
        config_file = "third_party/Entity/Entityv2/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml"

        self.entity_api = EntityApi(config_file=config_file)
        self.instance_score_thres = instance_score_thres
        self.inter_ratio_thres = inter_ratio_thres
        self.min_area_ratio_thres = min_area_ratio_thres

    def evaluate_on_sample(self, output_image: Image.Image, inpainting_mask: Image.Image):
        output_image = output_image.convert('RGB')
        output_image_np = np.array(output_image)[..., ::-1]
        predictions = self.entity_api(output_image_np)
        pred_masks = torch.nn.functional.one_hot(predictions['panoptic_seg'][0].long()).permute(2, 0, 1) > 0
        pred_masks = pred_masks[1:]
        if len(pred_masks) == 0:
            bad_case_mask = pred_masks
            pred_masks = pred_masks.new_zeros([1, *pred_masks.shape[1:]])
            bad_case_num = 0
        else:
            inpainting_mask = pil_to_tensor(inpainting_mask).cuda()
            inpainting_mask = inpainting_mask > 0
            inter_area = (pred_masks & inpainting_mask).sum((1, 2))
            inter_ratio_to_entity = inter_area / pred_masks.sum((1, 2)).clamp(min=1e-4)
            inter_ratio_to_inpainting = inter_area / inpainting_mask.sum((1, 2)).clamp(min=1e-4)
            bad_case = inter_ratio_to_entity > self.inter_ratio_thres
            bad_case = bad_case & (inter_ratio_to_inpainting > self.min_area_ratio_thres)
            bad_case_num = bad_case.sum()
            bad_case_num = bad_case_num.item()
            bad_case_mask = pred_masks[bad_case]
        if bad_case_num > 0:
            bad_case_inter_ratio_to_inpainting = inter_ratio_to_inpainting[bad_case].sum().item()
        else:
            bad_case_inter_ratio_to_inpainting = 0
        return DataDict(MSN=bad_case_num, MARS=bad_case_inter_ratio_to_inpainting, bad_case_mask=bad_case_mask, pred_masks=pred_masks, predictions=predictions)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path', type=str)
    parser.add_argument(
        '--show_path', type=str)
    parser.add_argument(
        '--data_indexes_path', type=str, default=None)
    parser.add_argument('--inter_ratio_thres', type=float, default=0.8)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    inter_ratio_thres = 0.95
    image_path = "unhcv/projects/diffusion/inpainting/evaluation/image.png"
    mask_path = "unhcv/projects/diffusion/inpainting/evaluation/mask.png"
    show_path = "/home/yixing/show/metric"
    removing_metric = RemovingMetric(inter_ratio_thres=inter_ratio_thres)
    result = obj_load(image_path)
    inpainting_mask = obj_load(mask_path)
    result = result.resize(inpainting_mask.size, resample=Image.BICUBIC)
    metric_data = removing_metric.evaluate_on_sample(result, inpainting_mask)
    print(f'MSN: {metric_data["MSN"]}, MARS: {metric_data["MARS"]}')