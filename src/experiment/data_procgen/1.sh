#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-procgen/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python ppo_procgen.py --track=True --env_id="climber"   --save_dir="../data/exp_icl//datasets//real/procgen/climber//"   &
CUDA_VISIBLE_DEVICES=1 python ppo_procgen.py --track=True --env_id="coinrun"   --save_dir="../data/exp_icl//datasets//real/procgen/coinrun//"   &
CUDA_VISIBLE_DEVICES=2 python ppo_procgen.py --track=True --env_id="dodgeball" --save_dir="../data/exp_icl//datasets//real/procgen/dodgeball//" &
CUDA_VISIBLE_DEVICES=3 python ppo_procgen.py --track=True --env_id="fruitbot"  --save_dir="../data/exp_icl//datasets//real/procgen/fruitbot//"  &
wait
