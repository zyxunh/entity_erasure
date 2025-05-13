set -x

MASTER_ADDR=127.0.0.3
MASTER_PORT=11310

accelerate launch --config_file \
${HOME}/code/unhcv/unhcv/core/train/accelerate_config.json \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  --num_machines 1 --num_processes 2 \
  train_mask2former.py \
  --train_batch_size=1 \
  --dataloader_num_workers=8 \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --checkpoint_root=${HOME}/train_outputs/checkpoint/unet_inpainting \
  --show_root=${HOME}/train_outputs/show/unet_inpainting \
  --save_steps=5000 \
  --test_steps=5000000 \
  --train_visual_steps=1000 \
  --train_steps=80000 \
  ${@:1}
