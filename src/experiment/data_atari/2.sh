#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-atari/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python ppo_atari_envpool.py --track=True --env_id="Riverraid-v5"     --save_dir="../data/exp_icl//datasets//real/atari/Riverraid//"     &
CUDA_VISIBLE_DEVICES=1 python ppo_atari_envpool.py --track=True --env_id="Hero-v5"          --save_dir="../data/exp_icl//datasets//real/atari/Hero//"          &
CUDA_VISIBLE_DEVICES=2 python ppo_atari_envpool.py --track=True --env_id="Krull-v5"         --save_dir="../data/exp_icl//datasets//real/atari/Krull//"         &
CUDA_VISIBLE_DEVICES=3 python ppo_atari_envpool.py --track=True --env_id="Tutankham-v5"     --save_dir="../data/exp_icl//datasets//real/atari/Tutankham//"     &
wait
