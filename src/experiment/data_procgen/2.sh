#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-procgen/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python ppo_procgen.py --track=True --env_id="heist"     --save_dir="../data/exp_icl//datasets//real/procgen/heist//"     &
CUDA_VISIBLE_DEVICES=1 python ppo_procgen.py --track=True --env_id="jumper"    --save_dir="../data/exp_icl//datasets//real/procgen/jumper//"    &
CUDA_VISIBLE_DEVICES=2 python ppo_procgen.py --track=True --env_id="leaper"    --save_dir="../data/exp_icl//datasets//real/procgen/leaper//"    &
CUDA_VISIBLE_DEVICES=3 python ppo_procgen.py --track=True --env_id="maze"      --save_dir="../data/exp_icl//datasets//real/procgen/maze//"      &
wait
