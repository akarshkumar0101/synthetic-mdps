#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-atari/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python ppo_atari_envpool.py --track=True --env_id="Kangaroo-v5"      --save_dir="../data/exp_icl//datasets//real/atari/Kangaroo//"      &
CUDA_VISIBLE_DEVICES=1 python ppo_atari_envpool.py --track=True --env_id="MsPacman-v5"      --save_dir="../data/exp_icl//datasets//real/atari/MsPacman//"      &
CUDA_VISIBLE_DEVICES=2 python ppo_atari_envpool.py --track=True --env_id="Defender-v5"      --save_dir="../data/exp_icl//datasets//real/atari/Defender//"      &
CUDA_VISIBLE_DEVICES=3 python ppo_atari_envpool.py --track=True --env_id="BeamRider-v5"     --save_dir="../data/exp_icl//datasets//real/atari/BeamRider//"     &
wait
