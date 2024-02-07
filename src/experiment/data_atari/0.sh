#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-atari/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python ppo_atari_envpool.py --track=True --env_id="Pong-v5"          --save_dir="../data/exp_icl//datasets//real/atari/Pong//"          &
CUDA_VISIBLE_DEVICES=1 python ppo_atari_envpool.py --track=True --env_id="Breakout-v5"      --save_dir="../data/exp_icl//datasets//real/atari/Breakout//"      &
CUDA_VISIBLE_DEVICES=2 python ppo_atari_envpool.py --track=True --env_id="SpaceInvaders-v5" --save_dir="../data/exp_icl//datasets//real/atari/SpaceInvaders//" &
CUDA_VISIBLE_DEVICES=3 python ppo_atari_envpool.py --track=True --env_id="Asterix-v5"       --save_dir="../data/exp_icl//datasets//real/atari/Asterix//"       &
wait
