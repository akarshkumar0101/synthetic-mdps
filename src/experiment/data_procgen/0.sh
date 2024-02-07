#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-procgen/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python ppo_procgen.py --track=True --env_id="bigfish"   --save_dir="../data/exp_icl//datasets//real/procgen/bigfish//"   &
CUDA_VISIBLE_DEVICES=1 python ppo_procgen.py --track=True --env_id="bossfight" --save_dir="../data/exp_icl//datasets//real/procgen/bossfight//" &
CUDA_VISIBLE_DEVICES=2 python ppo_procgen.py --track=True --env_id="caveflyer" --save_dir="../data/exp_icl//datasets//real/procgen/caveflyer//" &
CUDA_VISIBLE_DEVICES=3 python ppo_procgen.py --track=True --env_id="chaser"    --save_dir="../data/exp_icl//datasets//real/procgen/chaser//"    &
wait
