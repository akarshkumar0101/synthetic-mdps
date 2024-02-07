#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-procgen/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python ppo_procgen.py --track=True --env_id="miner"     --save_dir="../data/exp_icl//datasets//real/procgen/miner//"     &
CUDA_VISIBLE_DEVICES=1 python ppo_procgen.py --track=True --env_id="ninja"     --save_dir="../data/exp_icl//datasets//real/procgen/ninja//"     &
CUDA_VISIBLE_DEVICES=2 python ppo_procgen.py --track=True --env_id="plunder"   --save_dir="../data/exp_icl//datasets//real/procgen/plunder//"   &
CUDA_VISIBLE_DEVICES=3 python ppo_procgen.py --track=True --env_id="starpilot" --save_dir="../data/exp_icl//datasets//real/procgen/starpilot//" &
wait
