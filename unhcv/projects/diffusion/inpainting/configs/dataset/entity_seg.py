transforms_kwargs = dict(interpolations=("bicubic", "nearest-exact", "nearest-exact", "nearest-exact"), max_stride=64)
train_dataset_kwargs = dict(image_keys=('image', 'mask', 'thing_score_mask', "entity_proportion_mask"), transforms_kwargs=transforms_kwargs,
                            image_modes=("RGB", "L", "L", "L"),
                            backend_config=dict(decode_methods=dict(default=None)),
                            data_indexes_filter=None,
                            name_pair={"text": "text_blip-large"}, iou_filter_thres=0.6,
                            mask_generator_kwargs=dict(use_mask_step_id=True), default_text="empty",
                            collect_keys=('image', 'mask', 'inpainting_mask', 'text'),
                            remove_ratio_sub=dict(lama=0.25, merge=0.25, entity=0.5))

data_root = ["dataset/open-images-dataset/train/lmdb", "dataset/open-images-dataset/train/lmdb_1025"]
data_indexes_root = "dataset/open-images-dataset/train/lmdb_1025"
data_indexes_suffix = "_catalog.bson"
