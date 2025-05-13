import os
import os.path as osp
import random
import numpy as np
from typing import Sequence, Tuple
import PIL.Image
import PIL.Image as Image
import torch
from functools import wraps
from unhcv.common.utils import walk_all_files_with_suffix, find_path
from unhcv.common.utils import obj_load
from unhcv.common.types import DataDict
from unhcv.common.utils import get_logger
from unhcv.common.array import chunk, split
from unhcv.datasets.transforms.torchvision_transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomResizedWHCrop
from unhcv.common.image import area2hw
from torch.utils.data import DataLoader, Dataset as TDataset, IterableDataset as TIterableDataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import pil_to_tensor
from typing import List, Optional, Union
from line_profiler import profile
from unhcv.core.utils import get_global_size, get_global_rank
from .file_backend import FileBackend
from .lmdb_backend import LmdbBackend
try:
    from .kv_backend import obj_load_from_kv, obj_dump_to_kv, KvBackend
except (ModuleNotFoundError, ImportError):
    pass
from .size_bucket import SizeBucket

logger = get_logger(__name__)


class DatasetSharder:
    generator = None
    seed = 1234
    epoch = 0
    actual_id = 0
    actual_size = 1

    def set_seed(self):
        self.generator = torch.Generator().manual_seed(self.seed + self.epoch)
        # logger.info("actual_id is {}, set seed to {}".format(self.actual_id, self.seed + self.epoch))

    def __init__(self):
        self.set_seed()

    def new_epoch(self):
        self.epoch += 1
        self.set_seed()

    def set_epoch(self, epoch):
        logger.info("set epoch to {}".format(epoch))
        self.generator = torch.Generator().manual_seed(
            self.seed + epoch)

    def set_id(self, actual_id, actual_size):
        self.actual_id = actual_id
        self.actual_size = actual_size
        logger.info("actual_id is {}, actual_size is {}".format(actual_id, actual_size))


class Dataset(TIterableDataset):
    backend_type = "File"
    batched_record_index0_flag = False

    def __init__(self, data_indexes_path: Optional[str] = None, data_root: Optional[str] = None, root_ids=None,
                 collect_keys=None, shuffle=False, debug=False, batch_size=None, backend_config={},
                 parallel_read_num=1, data_indexes_filter=None, name_pair=None):
        super().__init__()
        if data_root is None:
            assert data_indexes_path.endswith("_catalog.bson")
            data_root = data_indexes_path[:-len("_catalog.bson")]
        data_indexes_path = find_path(data_indexes_path)
        self.data_indexes_path = data_indexes_path
        if data_indexes_path.startswith("hdfs"):
            self.data_indexes = self.data_information = obj_load_from_kv(data_indexes_path)
        else:
            self.data_indexes = self.data_information = obj_load(data_indexes_path)
        if 'data_root' in self.data_information:
            data_root = self.data_information['data_root']
        if isinstance(self.data_information, dict):
            self.backend_type = self.data_information.pop("backend_type", self.backend_type)
            self.data_indexes = self.data_information.pop("indexes")
        if root_ids is None and isinstance(self.data_information, dict):
            root_ids = self.data_information.get("root_ids", None)
        if data_indexes_filter is not None:
            self.data_indexes = data_indexes_filter(self.data_indexes)
        if self.backend_type == "File":
            self.data_backend = FileBackend(root=data_root, **backend_config)
        elif self.backend_type == "Lmdb":
            self.data_backend = LmdbBackend(root=data_root, root_ids=root_ids, **backend_config)
        elif self.backend_type == "Kv":
            self.data_backend = KvBackend(root=data_root, root_ids=root_ids, **backend_config)
        else:
            raise NotImplementedError

        self.data_root = data_root
        self.collect_keys = collect_keys
        self.shuffle = shuffle
        self.debug = debug
        self.batch_size = batch_size
        self.drop_last = True
        self.parallel_read_num = parallel_read_num
        self.name_pair = name_pair
        data_length = len(self)
        if batch_size is not None and self.drop_last:
            extra_num = data_length % self.batch_size
            if extra_num:
                logger.info("{} will drop {}".format(os.path.basename(data_indexes_path), extra_num))
        logger.info("{} length is {}".format(os.path.basename(data_indexes_path), data_length))

    def read_with_data_indexes(self, data_indexes, return_data_index=False):
        num_data_indexes = len(data_indexes)
        # data_index = self.data_indexes[index]

        index_for_recovery_list = []
        key_list = []
        value_list = []
        data = [{} for _ in range(num_data_indexes)]
        for i_data_index, data_index in enumerate(data_indexes):
            for key, value in data_index.items():
                if (isinstance(value, str) and 'text' not in key) or \
                        (isinstance(value, (Tuple, List)) and isinstance(value[0], str)):
                    key_list.append(key)
                    value_list.append(value)
                    index_for_recovery_list.append(i_data_index)
                else:
                    data[i_data_index][key] = value
        value_decoded_list = self.data_backend.read_many(value_list)
        for index_for_recovery, key, value_decoded in zip(index_for_recovery_list, key_list, value_decoded_list):
            if isinstance(value_decoded, dict):
                data[index_for_recovery].update(value_decoded)
            else:
                data[index_for_recovery][key] = value_decoded
        if self.name_pair is not None:
            for data_i in data:
                for name1, name2 in self.name_pair.items():
                    if name2 in data_i:
                        data_i[name1] = data_i[name2]
                        del data_i[name2]
        if return_data_index:
            return data, data_index
        return data

    @profile
    def read_with_index(self, index, return_data_index=False):
        if isinstance(index, int):
            index = (index, )
            index_int_flag = True
        else:
            index_int_flag = False
        data_indexes = [self.data_indexes[var] for var in index]
        data = self.read_with_data_indexes(data_indexes, return_data_index=return_data_index)
        if return_data_index:
            data, data_index = data
        if index_int_flag:
            data = data[0]
        if return_data_index:
            return data, data_index
        return data

    def postprocess(self, data):
        if self.debug:
            return data
        if self.collect_keys is not None:
            new_data = {}
            for key in self.collect_keys:
                new_data[key] = data[key]
            data = new_data
        return data

    def __len__(self):
        return len(self.data_indexes)

    def __getitem__(self, index):
        return self.read_with_index(index)

    def batched_record_index0_start(self):
        self.batched_record_index0_flag = True

    def batched_record_index0_end(self):
        self.batched_record_index0_flag = False

    @profile
    def __iter__(self):
        data_length = len(self)
        if self.shuffle:
            indexes = np.random.permutation(data_length)
        else:
            indexes = np.arange(data_length)

        if self.batch_size is not None:
            # assert self.batch_size < data_length
            if self.batch_size > data_length:
                print(
                    "data_root {}'s length is {} batch_size is {}".format(self.data_root, data_length, self.batch_size))
                return
            if self.drop_last:
                extra_num = data_length % self.batch_size
                if extra_num:
                    random_indexes = np.random.permutation(data_length)
                    indexes = np.delete(indexes, random_indexes[:extra_num])

            assert len(indexes) % self.batch_size == 0
            indexes_chunked = chunk(indexes, self.batch_size)
        else:
            raise ValueError("Batch size must be number")
        for indexes in indexes_chunked:
            if self.batch_size is not None:
                self.batched_record_index0_start()
            indexes = chunk(indexes, self.parallel_read_num)
            for i_index, index in enumerate(indexes):
                if self.debug:
                    datas = self.read_with_index(index)
                else:
                    datas = self.read_with_index(index)
                for data in datas:
                    data = self.postprocess(data)
                    yield data

        # for index in indexes:
        #     data = self.read_with_index(index)
        #     yield data


class DatasetWithPreprocess(Dataset):
    def __init__(self, data_indexes_path: Optional[str] = None, data_root: Optional[str] = None,
                 transforms_kwargs=dict(), image_keys=("image", "mask"), image_modes=None, collect_keys=None,
                 shuffle=False, debug=False, batch_size=None, backend_config={}, parallel_read_num=None,
                 data_indexes_filter=None, name_pair=None, inference=False):
        if isinstance(image_keys, str):
            image_keys = (image_keys,)
        self.image_keys = image_keys
        self.image_modes = image_modes
        self.transform_method = "default"
        self.inference = inference
        super().__init__(data_indexes_path=data_indexes_path, data_root=data_root, collect_keys=collect_keys,
                         shuffle=shuffle, debug=debug, batch_size=batch_size, backend_config=backend_config,
                         parallel_read_num=parallel_read_num, data_indexes_filter=data_indexes_filter, name_pair=name_pair)
        self.init_transforms(transforms_kwargs)

    def build_bucket_shape_transforms(self, image):
        hw = self.size_bucket.match((image.height, image.width))
        # hw_ratio = data['image'].height / data['image'].width
        # hw = area2hw(area=self.transform_max_area, hw_ratio=hw_ratio, max_stride=self.max_stride)
        ratio = image.width / image.height
        self.random_resized_crop = RandomResizedCrop(size=hw, ratio=(ratio, ratio), scale=(1, 1),
                                                     interpolations=self.interpolations)

    @profile
    def preprocess(self, data):
        image_dict = {}
        images: List[PIL.Image.Image] = []
        for key in self.image_keys:
            images.append(data[key])
        image_modes = self.image_modes
        if image_modes is not None:
            if isinstance(image_modes, str):
                image_modes = [image_modes] * len(images)
            _images = []
            for image, image_mode in zip(images, image_modes):
                if image.mode != image_mode:
                    if (image.mode == "RGBA" or image.info.get("transparency",
                                                               None) is not None) and image_mode == "RGB":
                        image = image.convert("RGBA")
                        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                        white.paste(image, mask=image.split()[3])
                        image = white
                    else:
                        image = image.convert(image_mode)
                _images.append(image)
            images = _images
        if self.batched_resize and self.batched_record_index0_flag:
            self.batched_record_index0_end()
            self.build_bucket_shape_transforms(data['image'])
        if self.transform_method == "default":
            pass
        elif self.transform_method == "resize_to_bucket_shape":
            self.build_bucket_shape_transforms(data['image'])
        if self.inference:
            image_dict['original_hw'] = (images[0].height, images[0].width)
        images_value = self.random_resized_crop(list(images), self.interpolations)
        image_dict.update(dict(zip(self.image_keys, images_value)))
        return image_dict

    def init_transforms(self, transforms_kwargs):
        interpolations = transforms_kwargs.get("interpolations", ("bicubic", "nearest-exact"))
        if isinstance(interpolations, str):
            interpolations = (interpolations, )
        self.interpolations = interpolations
        self.batched_resize = transforms_kwargs.get("batched_resize", True)
        self.transform_max_area = transforms_kwargs.get("max_area", 512 * 512)
        self.max_stride = transforms_kwargs.get("max_stride", 1)
        size = transforms_kwargs.get("size", 512)
        scale = transforms_kwargs.get("scale", (0.9, 1))
        self.transform_method = transforms_kwargs.get("transform_method", "default")
        self.random_resized_crop = RandomResizedWHCrop(size=size, scale=scale, ratio=(1, 1), interpolations=interpolations)
        # self.random_flip = RandomHorizontalFlip()
        if self.batched_resize:
            self.size_bucket = SizeBucket(stride=self.max_stride, **transforms_kwargs.get("size_bucket_config", {}))

    @profile
    def read_with_index(self, index):
        if isinstance(index, int):
            index = (index,)
            index_int_flag = True
        else:
            index_int_flag = False
        data = super().read_with_index(index)
        if not isinstance(data, list):
            data = [data]

        for data_i in data:
            image_dict = self.preprocess(data_i)
            if self.debug:
                for key in image_dict.keys():
                    if key in data_i:
                        data_i[key + "_raw"] = data_i[key]
            data_i.update(image_dict)
        if index_int_flag:
            data = data[0]
        return data

    def __getitem__(self, index):
        data = self.read_with_index(index)
        data = self.postprocess(data)
        return data

    def postprocess(self, data):
        for key, value in data.items():
            if isinstance(value, PIL.Image.Image):
                data[key] = pil_to_tensor(value)
        return super().postprocess(data)


class ConcatDataset(TIterableDataset, DatasetSharder):

    def __init__(self, dataset_clses, common_config=None, custom_configs=None, read_mode="one_by_one", choice_p=None,
                 max_per_read=None, infinite=True):
        DatasetSharder.__init__(self)
        self.custom_configs = custom_configs
        self.dataset_clses = dataset_clses
        self.common_config = common_config
        self.read_mode = read_mode
        self.choice_p = choice_p
        self.datasets = []
        self.per_dataset_length = []
        self.max_per_read = max_per_read
        self.infinite = infinite

        if isinstance(dataset_clses, tuple):
            for dataset in dataset_clses:
                self.datasets.append([dataset, iter(dataset)])
                self.per_dataset_length.append(len(dataset))
        else:
            self.num_dataset = len(custom_configs)
            if read_mode == "random_choice":
                for i_dataset in range(len(custom_configs)):
                    dataset_cls = self.dataset_clses[i_dataset] if isinstance(self.dataset_clses,
                                                                              tuple) else self.dataset_clses
                    dataset = dataset_cls(**self.common_config, **self.custom_configs[i_dataset])
                    self.datasets.append([dataset, iter(dataset)])
                    self.per_dataset_length.append(len(dataset))
            elif read_mode == "one_by_one":
                self.per_dataset_length = [var.pop("length") for var in custom_configs]
        logger.info("num_dataset is {}, len_dataset is {}".format(self.num_dataset, len(self)))

    def __len__(self):
        return sum(self.per_dataset_length)

    def __iter__(self):
        if self.read_mode == "one_by_one":
            while True:
                randperm_indexes = torch.randperm(self.num_dataset, generator=self.generator)
                logger.info(
                    f"actual_id is {self.actual_id}, epoch is {self.epoch}, randperm_indexes first 5 {randperm_indexes[:5]}")
                randperm_indexes = split(randperm_indexes, self.actual_size)[self.actual_id]
                # logger.info(f"sharder id: {self.actual_id}, size: {self.actual_size}")
                for i_dataset in randperm_indexes:
                    dataset_cls = self.dataset_clses[i_dataset] if isinstance(self.dataset_clses, tuple) else self.dataset_clses
                    dataset = dataset_cls(**self.common_config, **self.custom_configs[i_dataset])
                    readed_num = 0
                    try:
                        dataset_iter = iter(dataset)
                        while True:
                            try:
                                data = next(dataset_iter)
                            except StopIteration:
                                break
                            yield data
                            readed_num += 1
                            if self.max_per_read is not None and readed_num >= self.max_per_read:
                                break
                    except Exception as ex:
                        logger.error("data_root {} read error, error is {}".format(
                            self.custom_configs[i_dataset]["data_root"], ex))
                        import traceback
                        traceback.print_exc()

                self.new_epoch()
                if not self.infinite:
                    break

        elif self.read_mode == "random_choice":
            raise NotImplementedError
            dataset = np.random.choice(self.datasets, self.choice_p)
            try:
                data = next(dataset[1])
            except StopIteration:
                dataset[1] = iter(dataset[0])
                data = next(dataset[1])
            yield data
        else:
            raise NotImplementedError

    def set_id(self, actual_id, actual_size):
        super().set_id(actual_id, actual_size)


def sharder_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    global_size = get_global_size()
    global_rank = get_global_rank()
    num_workers = worker_info.num_workers
    actual_size = global_size * num_workers
    actual_id = global_rank * num_workers + worker_id
    if isinstance(dataset, DatasetSharder):
        dataset.set_id(actual_id=actual_id, actual_size=actual_size)


if __name__ == "__main__":
    pass
    # from unhcv.common.utils import obj_dump
    # from unhcv.common.utils import write_im
    # from unhcv.common.image import visual_mask, putText, concat_differ_size
    #
    # data_root = "hdfs://haruna/dp/mloops/datasets/unh/dataset/openimage/entity_seg_170w_caption_debug/000000"
    # data_indexes_path = "hdfs://haruna/dp/mloops/datasets/unh/dataset/openimage/entity_seg_170w_caption_debug/000000_catalog.msgpack"
    # show_root = "/home/tiger/show"
    # data_root = "hdfs://haruna/dp/mloops/datasets/unh/dataset/openimage/entity_seg_170w_caption_debug/005_00010"
    # data_indexes_path = "hdfs://haruna/dp/mloops/datasets/unh/dataset/openimage/entity_seg_170w_caption_debug/005_00010_catalog.bson"
    # indexes = obj_load_from_kv(data_indexes_path)
    # breakpoint()
    # show_root = os.path.join(show_root, os.path.basename(data_root))
    #
    # dataset = Dataset(data_indexes_path=data_indexes_path, data_root=data_root)
    # for i in range(len(dataset)):
    #     data = dataset.read_with_index(i)
    #     image, mask = np.array(data['image'].convert('RGB'))[..., ::-1], np.array(data['mask'])
    #     text = putText(np.zeros_like(image), show_texts=data['text_blip-large'])
    #     show = concat_differ_size([image, mask, text], axis=1)
    #     write_im(osp.join(show_root, f"{i}.jpg"), show)
    #     # image = visual_mask([..., ::-1], np.array(data['mask']))[-1]
    # exit()
    #
    # file_name = "/home/tiger/dataset/entity_openimage_mdb/indexes/train_image_indexes.msgpack"
    # indexes = obj_load(file_name)
    # breakpoint()
    # indexes.update(data_backend_type="Lmdb", subdirs=["img_mdb", "pseudo_mdb"])
    # obj_dump(file_name, indexes)
    # exit()
    # breakpoint()
    # pass
    # from unhcv.datasets.utils.torch_dataset_wrap import wrap_torch_dataset
    # from unhcv.projects.diffusion.ldm import multi_patch_noise
    # from unhcv.common.image.visual import imwrite_tensor
    #
    # dataset = DatasetWithPreprocess(data_indexes_path="/home/tiger/dataset/entity_openimage_mdb/indexes/test_image_indexes.yml",
    #                                 data_root="/home/tiger/dataset/entity_openimage_mdb", image_keys=("image", "mask"),
    #                                 transforms_kwargs=dict(scale=(1, 1)))
    # data = dataset.read_with_index(0)
    # breakpoint()
    # dataset = DatasetWithPreprocess(data_indexes_path="/home/tiger/dataset/Adobe_EntitySeg/image_indexes/train_lr_indexes.yml", data_root="/home/tiger/dataset/Adobe_EntitySeg")
    # dataset.read_with_index(0)
    # breakpoint()
    # segmentation_dataset = wrap_torch_dataset(
    #     SegmentationDataset(image_roots=["/home/tiger/dataset/ADEChallengeData2016/images/training",
    #                                      "/home/tiger/dataset/ADEChallengeData2016/annotations_rgb/training"],
    #                         image_type_names=["image", "segmentation"]))
    # dataloader = torch.utils.data.DataLoader(segmentation_dataset, num_workers=1, batch_size=2)
    # dataloader_iter = iter(dataloader)
    # out = next(dataloader_iter)
    # noise = multi_patch_noise(out['segmentation'].shape, (1, 3, 5, 7))
    # noise_segmentation = out['segmentation'] * 0.2 + noise * 0.8
    # imwrite_tensor("/home/tiger/code/unhcv/show.jpg", noise_segmentation)
    # noise = multi_patch_noise(out['segmentation'].shape, (1,))
    # noise_segmentation = out['segmentation'] * 0.2 + noise * 0.8
    # imwrite_tensor("/home/tiger/code/unhcv/show1.jpg", noise_segmentation)
    # breakpoint()
    # out = segmentation_dataset.read_i(0)
    # segmentation_dataset_iter = iter(segmentation_dataset)
    # out = next(segmentation_dataset_iter)
    # breakpoint()
    # pass
