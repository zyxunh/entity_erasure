set -x

MASTER_ADDR=127.0.0.3
MASTER_PORT=11312
NUM_PROCESS=$1


accelerate launch --config_file \
${HOME}/code/unhcv/unhcv/core/train/deep_speed_bf16.yml \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
  --num_machines 1 --num_processes $NUM_PROCESS \
  train_inpainting.py \
  --train_batch_size=1 \
  --dataloader_num_workers=8 \
  --learning_rate=5e-05 \
  --weight_decay=0.01 \
  --checkpoint_root=${HOME}/train_outputs/checkpoint/unet_inpainting \
  --show_root=${HOME}/train_outputs/show/unet_inpainting \
  --save_steps=2500 \
  --test_steps=1000000 \
  --train_visual_steps=125 \
  --train_steps=20000 \
  ${@:2}

