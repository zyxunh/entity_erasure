import os
import argparse
import time

import torch.distributed
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from typing import Any, Optional, Union, Dict, List, Sequence

import torch
import torch.nn as nn
import numpy as np
import wandb
from diffusers.optimization import get_scheduler

from unhcv.common.image import pad_image_to_same_size
from unhcv.common.utils.progressbar import ProgressBarTqdm
from unhcv.common.types import DataDict
from unhcv.common.utils import obj_dump, obj_load, write_txt, human_format_num, MeanCache
from unhcv.common.fileio.hdfs import copy, remove_dir, listdir
from unhcv.datasets.utils.test_data import get_train2test_iterabledataloader


class AccelerateModelWrap:

    @property
    def trained_models(self):
        raise NotImplementedError

    @property
    def frozen_models(self):
        raise NotImplementedError

    def reset_trained_models(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AccelerateTrain:
    default_dataset_kwargs: Optional[Union[Dict, Sequence[Dict]]] = None
    default_demo_dataset_kwargs: Optional[Dict] = None
    dataset_class: Optional[type] = None
    frozen_models: Optional[List[nn.Module]] = []
    global_step: int = 1
    save_for_training_show_tensors: Optional[DataDict] = None
    max_visual_num_in_training: int = 5
    demo_dataset_class: Optional[type] = None
    accelerator: Optional[Accelerator] = None
    state_dict_save_path_queue: Dict[str, List[str]] = dict(local=[], hdfs=[])
    log_cache_memory = {}
    last_save_state_step = -1
    mean_cache = MeanCache()
    proc_asynchronization = []
    prepare_train_dataloader = True

    def __init__(self, *args, **kwargs) -> None:
        self.model: Optional[nn.Module] = None
        self.logger = None
        self.args: Optional[argparse.Namespace] = None
        self.train_dataiter: Optional[iter] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.weight_dtype: Optional[torch.dtype] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.demo_dataset: Optional[Dataset] = None
        self.demo_dataloader: Optional[DataLoader] = None
        self.init_log()
        self.get_args()
        self.init_accelerator()
        if not self.args.eval_only:
            self.init_train_dataset()
        if self.args.debug:
            if self.args.debug_mode == "dataset":
                self.debug_for_dataset()
                self.accelerator.wait_for_everyone()
                exit()
            elif self.args.debug_mode == "demo":
                self.init_for_demo()
                self.debug_for_demo()
                exit()
            else:
                pass
        self.init_model()
        self.froze_model()
        self.trained_parameter_names = trained_parameter_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trained_parameter_names.append(name)
        if self.args.eval_only:
            self.init_for_eval()
        else:
            self.init_for_train()
        self.save_project_information()

    def parser_add_argument(self):
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument(
            "--checkpoint_root",
            type=str,
            default="/home/tiger/train_outputs/checkpoint"
        )
        parser.add_argument(
            "--show_root",
            type=str,
            default="/home/tiger/train_outputs/show"
        )
        parser.add_argument(
            "--hdfs_root",
            type=str,
            default=None
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--show_dir",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--hdfs_dir",
            type=str,
            default=None,
        )
        parser.add_argument("--project_suffix", type=str, default=None)
        parser.add_argument("--max_local_state_num", type=int, default=1)
        parser.add_argument("--max_hdfs_state_num", type=int, default=3)
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-4,
            help="Learning rate to use.",
        )
        parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="constant",
            help=(
                'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                ' "constant", "constant_with_warmup"]'
            ),
        )
        parser.add_argument(
            "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
        )
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument("--max_grad_norm", type=float, default=None)

        parser.add_argument("--num_train_epochs", type=int, default=100)
        parser.add_argument(
            "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            type=int,
            default=0,
            help=(
                "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
            ),
        )
        parser.add_argument(
            "--save_steps",
            type=int,
            default=2000,
            help=(
                "Save a checkpoint of the training state every X updates"
            ),
        )
        parser.add_argument(
            "--test_steps",
            type=int,
            default=1000
        )
        parser.add_argument(
            "--train_steps",
            type=int,
            default=100000
        )
        parser.add_argument(
            "--train_visual_steps",
            type=int,
            default=None
        )
        parser.add_argument(
            "--mixed_precision",
            type=str,
            default=None,
            choices=["no", "fp16", "bf16"],
            help=(
                "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
            ),
        )
        parser.add_argument(
            "--report_to",
            type=str,
            default="tensorboard",
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
            ),
        )
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        parser.add_argument("--extra_checkpoint", type=str, default=None)
        parser.add_argument("--checkpoint", type=str, default=None)
        parser.add_argument("--eval_only", action="store_true")
        parser.add_argument("--eval_before_train", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--debug_mode", type=str, default="inference")
        parser.add_argument("--clip_image_size", type=int, default=224)
        parser.add_argument(
            "--project_name",
            type=str,
            default=None,
        )
        parser.add_argument("--dataset_config", type=str, default=None)
        parser.add_argument("--model_config", type=str, default=None)
        parser.add_argument("--args_file", type=str, default=None)
        parser.add_argument("--demo_dir", type=str, default=None)
        return parser

    def get_args(self):
        parser = self.parser_add_argument()
        args = parser.parse_args()
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if args.args_file is not None:
            args_loaded = obj_load(args.args_file)["updated_args"]
            for key in args.__dict__.keys():
                if getattr(args, key) == parser.get_default(key):
                    setattr(args, key, getattr(args_loaded, key))
        if args.project_name is None:
            if args.model_config is not None:
                args.project_name = os.path.basename(os.path.splitext(args.model_config)[0])
            else:
                args.project_name = "null"
            if args.dataset_config is not None:
                args.project_name += '_' + os.path.basename(os.path.splitext(args.dataset_config)[0])
        if args.project_suffix is not None:
            args.project_name += "_" + args.project_suffix
        if args.checkpoint_dir is None:
            args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
        if args.show_dir is None:
            args.show_dir = os.path.join(args.show_root, args.project_name)
        if args.hdfs_dir is None and args.hdfs_root is not None:
            args.hdfs_dir = os.path.join(args.hdfs_root, args.project_name)
        if args.train_visual_steps is None:
            args.train_visual_steps = args.test_steps
        if args.debug:
            args.checkpoint_dir += "_debug"
        if env_local_rank == -1 or env_local_rank == 0:
            if args.debug:
                remove_dir(args.checkpoint_dir)
        if env_local_rank != -1:
            args.local_rank = env_local_rank
        self.args = args
        if env_local_rank == -1 or env_local_rank == 0:
            default_and_updated_args = dict(default_args={}, updated_args={})
            for key, value in args.__dict__.items():
                if value == parser.get_default(key):
                    default_and_updated_args["default_args"][key] = value
                else:
                    default_and_updated_args["updated_args"][key] = value
            obj_dump(os.path.join(args.checkpoint_dir, 'information', 'args.yml'),
                     default_and_updated_args)

    def init_log(self):
        import logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger = get_logger(__name__)
        self.logger = logger

    def init_model(self):
        args = self.args
        accelerator = self.accelerator
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

    def froze_model(self):
        for model in self.frozen_models:
            if model is not None:
                model.requires_grad_(False)

    @property
    def trained_parameters(self):
        return self.model.parameters()

    def parse_model_config(self):
        return obj_load(self.args.model_config) if self.args.model_config else {}

    def parse_dataset_config(self):
        return obj_load(self.args.dataset_config) if self.args.dataset_config else {}

    def init_train_dataset(self):
        args = self.args
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        dataset_config = self.parse_dataset_config()
        if isinstance(self.default_dataset_kwargs, dict):
            default_dataset_kwargs = [self.default_dataset_kwargs]
        else:
            default_dataset_kwargs = self.default_dataset_kwargs
        train_datasets = []
        for dataset_kwargs in default_dataset_kwargs:
            dataset_kwargs = dataset_kwargs.copy()
            dataset_kwargs.update(dataset_config.get("train_dataset_kwargs", {}))
            dataset_kwargs["batch_size"] = args.train_batch_size
            train_dataset: Dataset = self.dataset_class(**dataset_kwargs)
            train_datasets.append(train_dataset)
        if len(train_datasets) > 1:
            train_dataset = ConcatDataset(train_datasets)
        else:
            train_dataset = train_datasets[0]
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            prefetch_factor=8 if args.dataloader_num_workers > 0 else None,
            collate_fn=torch.utils.data.default_collate,
            pin_memory=True,
            shuffle=True,
            drop_last=True)

        self.logger.info("***** Running training *****")
        self.logger.info(
            f"world_size is {world_size}, batch_size is {world_size * args.train_batch_size * args.gradient_accumulation_steps}, "
            f"num_epoch is {world_size * args.train_batch_size * args.train_steps * args.gradient_accumulation_steps / len(train_dataset)}")
        time.sleep(10)
        self.train_dataloader = train_dataloader

    def init_test_dataset(self, test_for_which="train", test_num=50):
        if test_for_which == "train":
            default_dataset_kwargs = self.default_dataset_kwargs.copy()
            dataset_config = obj_load(self.args.dataset_config) if self.args.dataset_config else {}
            default_dataset_kwargs.update(dataset_config.get("train_dataset_kwargs", {}))
            default_dataset_kwargs["batch_size"] = 1
            train_dataset = clip_dataset = ClipDataset(**default_dataset_kwargs)
            val_dataloader = get_train2test_iterabledataloader(train_dataset, num=test_num)
        return val_dataloader

    def init_for_train(self):
        args = self.args
        accelerator = self.accelerator
        model = self.model
        optimizer = torch.optim.AdamW(self.trained_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
        # Prepare everything with our `accelerator`.
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.train_steps * accelerator.num_processes,
        )
        if self.prepare_train_dataloader:
            model, optimizer, self.train_dataloader, self.lr_scheduler = accelerator.prepare(model, optimizer, self.train_dataloader, lr_scheduler)
        else:
            model, optimizer, self.lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        train_dataiter = iter(self.train_dataloader)
        self.train_dataiter = train_dataiter
        self.model = model
        self.optimizer = optimizer

    def init_for_eval(self):
        raise NotImplementedError

    def init_for_demo(self):
        if self.default_demo_dataset_kwargs is not None:
            default_demo_dataset_kwargs = self.default_demo_dataset_kwargs.copy()
            default_demo_dataset_kwargs["batch_size"] = 1
            dataset_config = self.parse_dataset_config()
            default_demo_dataset_kwargs.update(dataset_config.pop("demo_dataset_kwargs", {}))
            self.demo_dataset = self.demo_dataset_class(**default_demo_dataset_kwargs)
            self.demo_dataloader = DataLoader(
                dataset=self.demo_dataset,
                batch_size=1,
                num_workers=0,
                collate_fn=torch.utils.data.default_collate,
                pin_memory=True,
                shuffle=False)

    def backward(self, loss_dict):
        losses = []
        for key, value in loss_dict.items():
            if value.requires_grad:
                losses.append(value)
            loss_dict[key] = value.detach()
        losses = sum(losses)
        self.accelerator.backward(losses)
        return loss_dict

    def get_loss(self, batch: Any):
        raise NotImplementedError

    def inference_on_train(self, global_step):
        pass

    def inference_on_test(self, global_step):
        pass

    def inference_on_demo(self, global_step):
        pass

    def visual_training_result(self):
        pass

    def log_images_to_cache(self, images, tag="image", captions=None, reverse_color=False):
        if not isinstance(images, Sequence):
            images = [images]
        if reverse_color:
            images = [var[..., ::-1] for var in images]
        assert captions is None
        memory = self.log_cache_memory.get(tag, [])
        memory.extend(images)
        self.log_cache_memory[tag] = memory

    def log_images_cache_push(self):
        for tag, images in self.log_cache_memory.items():
            self.log_images(images, tag=tag)
        self.log_cache_memory.clear()

    def log_images(self, images, tag="image", captions=None, reverse_color=False):
        if not isinstance(images, Sequence):
            images = [images]
        images = pad_image_to_same_size(images)
        if reverse_color:
            images = [var[..., ::-1] for var in images]
        if captions is None:
            captions = ["{:3}".format(i) for i in range(len(images))]
        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images: np.ndarray = np.stack([np.asarray(img) for img in images])
                if np_images.ndim == 3:
                    np_images = np_images[..., None]
                tracker.writer.add_images(tag, np_images, self.global_step, dataformats="NHWC")
            elif tracker.name == "wandb":
                tracker.log(
                    {
                        tag: [
                            wandb.Image(image, caption=f"{captions[i]}")
                            for i, image in enumerate(images)
                        ]
                    }, step=self.global_step
                )
            else:
                self.logger.warning(f"image logging not implemented for {tracker.name}")

    def debug_for_dataset(self):
        for _ in range(100):
            data = next(self.train_dataiter)

    def debug_for_demo(self):
        show_root = os.path.join(self.args.show_dir, 'demo_dataset')
        remove_dir(show_root)
        for i_data, data in enumerate(self.demo_dataloader):
            pass

    def inference(self, global_step):
        self.init_for_demo()
        if self.demo_dataloader is not None:
            self.inference_on_demo(global_step=global_step)
            self.accelerator.wait_for_everyone()
        self.inference_on_test(global_step=global_step)
        self.accelerator.wait_for_everyone()
        self.inference_on_train(global_step=global_step)
        self.accelerator.wait_for_everyone()

    def init_accelerator(self):
        args = self.args
        logging_dir = os.path.join(args.show_dir, "log")
        accelerator_project_config = ProjectConfiguration(project_dir=args.checkpoint_dir, logging_dir=logging_dir)
        self.accelerator = accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

        if accelerator.distributed_type == accelerate.utils.DistributedType.DEEPSPEED:
            # accelerator.state.deepspeed_plugin.deepspeed_config[
            #     "gradient_accumulation_steps"] = args.gradient_accumulation_steps
            accelerator.state.deepspeed_plugin.deepspeed_config[
                'train_micro_batch_size_per_gpu'] = args.train_batch_size

        if self.accelerator.is_main_process:
            if args.report_to == "tensorboard":
                remove_dir(logging_dir)
            os.makedirs(logging_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = logging_dir
            accelerator.init_trackers(args.project_name, config=dict(vars(args)))

    @staticmethod
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = accelerate.state.AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    def save_model_information(self, model, mode_name):
        if model is None:
            return
        write_txt(
            os.path.join(self.args.checkpoint_dir, 'information', f'{mode_name}.txt'),
            model.__str__())
        trained_parameters = []
        trained_parameters_num = 0
        frozen_parameters_num = 0
        frozen_parameters = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trained_parameters_num += param.numel()
                trained_parameters.append(f"{name}: {list(param.shape)}\n")
            else:
                frozen_parameters_num += param.numel()
                frozen_parameters.append(f"{name}: {list(param.shape)}\n")

        write_txt(
            os.path.join(self.args.checkpoint_dir, 'information',
                         f'{mode_name}_trained_parameters.txt'), trained_parameters)
        write_txt(
            os.path.join(self.args.checkpoint_dir, 'information',
                         f'{mode_name}_frozen_parameters.txt'), frozen_parameters)
        write_txt(
            os.path.join(self.args.checkpoint_dir, 'information',
                         f'{mode_name}_parameters_num.txt'),
            f"trained_parameters_num: {human_format_num(trained_parameters_num)}\nfrozen_parameters_num: {human_format_num(frozen_parameters_num)}"
        )
        if self.args.hdfs_dir is not None:
            for file in listdir(self.args.checkpoint_dir):
                copy(file, self.args.hdfs_dir)

    def save_project_information(self):
        if self.accelerator.is_main_process:
            self.save_model_information(self.model, mode_name="model_0")
            for i_model, model in enumerate(self.frozen_models):
                self.save_model_information(model, mode_name=f"model_{i_model + 1}")
        self.accelerator.wait_for_everyone()

    def save_state(self):
        if self.last_save_state_step != self.global_step:
            self.last_save_state_step = self.global_step
        else:
            return
        save_path = os.path.join(self.args.checkpoint_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            files = listdir(save_path)
            for file in files:
                if os.path.basename(file) != "model.bin":
                    remove_dir(file)
            if self.args.max_local_state_num is not None:
                self.state_dict_save_path_queue['local'].append(save_path)
                if len(self.state_dict_save_path_queue['local']) > self.args.max_local_state_num:
                    remove_dir(self.state_dict_save_path_queue['local'][0])
                    del self.state_dict_save_path_queue['local'][0]
            if self.args.hdfs_dir is not None:
                hdfs_save_path = os.path.join(self.args.hdfs_dir, f"checkpoint-{self.global_step}")
                self.proc_asynchronization.append(copy(save_path, hdfs_save_path, asynchronization=True))
                if self.args.max_hdfs_state_num is not None:
                    self.state_dict_save_path_queue['hdfs'].append(hdfs_save_path)
                    if len(self.state_dict_save_path_queue['hdfs']) > self.args.max_hdfs_state_num:
                        remove_dir(self.state_dict_save_path_queue['hdfs'][0])
                        del self.state_dict_save_path_queue['hdfs'][0]
        self.accelerator.wait_for_everyone()

    def main(self):
        if self.args.eval_only:
            self.inference(global_step=0)
        else:
            if self.args.debug:
                self.inference(global_step=0)
            self.train()


    def train(self):
        print("start train")
        args = self.args
        accelerator = self.accelerator

        if accelerator.is_main_process and args.checkpoint_dir is not None:
            os.makedirs(args.checkpoint_dir, exist_ok=True)

        def save_model_hook(models, weights, output_dir):
            for i, model_to_save in enumerate(models):
                # model_to_save.save_pretrained(os.path.join(output_dir, "unet"), is_main_process=accelerator.is_main_process)
                save_directory = output_dir
                os.makedirs(save_directory, exist_ok=True)
                # Save the model
                # FSDP get state_dict need all the process
                _state_dict: Dict[str, Any] = model_to_save.state_dict()
                requires_grad_keys = []
                for key, para in model_to_save.named_parameters():
                    if para.requires_grad:
                        requires_grad_keys.append(key)
                requires_grad_keys = set(requires_grad_keys)
                state_dict = {}
                for key, value in _state_dict.items():
                    if key in requires_grad_keys:
                        if key.startswith("module."):
                            key = key[len("module."):]
                        state_dict[key] = value
                weights_name = 'model.bin'
                if accelerator.is_main_process:
                    torch.save(state_dict, os.path.join(save_directory, weights_name))
                    print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")
            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        # self.init_model()
        # self.init_for_train()
        train_steps = args.train_steps
        if args.eval_only:
            return
        if args.eval_before_train:
            self.inference(global_step=self.global_step)
        progress_bar = ProgressBarTqdm(
            train_steps, disable=not accelerator.is_local_main_process, smoothing=0)
        while True:
            begin = time.perf_counter()
            try:
                batch: Any = next(self.train_dataiter)
            except StopIteration:
                self.train_dataiter = iter(self.train_dataloader)
                batch: Any = next(self.train_dataiter)
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(self.model):
                # Convert images to latent space
                loss_dict = self.get_loss(batch)
                # Gather the losses across all processes for logging (if we use distributed training).
                loss_for_backward = sum(loss_dict.values())
                for key, value in loss_dict.items():
                    loss_dict[key] = accelerator.reduce(value, "mean").item()
                loss_sum = accelerator.reduce(loss_for_backward, "mean").item()
                self.mean_cache.update(dict(loss=loss_sum))
                self.mean_cache.update(loss_dict)
                # Backward
                accelerator.backward(loss_for_backward)
                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    grad_before_norm = accelerator.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    self.mean_cache.update(dict(grad=grad_before_norm.item()))

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            if accelerator.sync_gradients:
                self.global_step += 1
                tacker_logs = {}
                mean_dict = self.mean_cache.mean()
                tacker_logs.update(mean_dict)
                for i_lr, lr in enumerate(self.lr_scheduler.get_lr()):
                    tacker_logs[f"lr_{i_lr}"] = lr
                accelerator.log(tacker_logs, step=self.global_step)
                progress_bar_logs = dict(loss=mean_dict.get('loss'))
                grad = mean_dict.get("grad", None)
                if grad is not None:
                    progress_bar_logs["grad"] = grad

                progress_bar.log(progress_bar_logs)  # , "grad": grad.item()
                progress_bar.update()

                if self.global_step % args.save_steps == 0:
                    self.save_state()
                if self.global_step >= train_steps:
                    break
                if self.global_step % args.test_steps == 0:
                    self.inference(global_step=self.global_step)
                if self.save_for_training_show_tensors is not None:
                    if self.accelerator.is_main_process:
                        self.visual_training_result()
                    self.save_for_training_show_tensors = None
                    self.accelerator.wait_for_everyone()

        # save_path = os.path.join(args.checkpoint_dir, f"checkpoint-{self.global_step}")
        self.save_state()
        # if not os.path.exists(save_path):
        #     accelerator.save_state(save_path)
        self.inference(global_step=self.global_step)
        accelerator.end_training()

    def end_training(self):
        for proc in self.proc_asynchronization:
            stdout, stderr = proc.communicate()
            if stdout is not None or stderr is not None:
                print(f'stdout is {stdout}, strderr is {stderr}')
        self.accelerator.end_training()


if __name__ == "__main__":
    pass
