python icl_gen_ed.py --env_id="name=Asterix-MinAtar;tl=500"       --agent_id="minatar" --n_seeds_seq=1 --n_seeds_par=1 --n_envs=128 --n_envs_batch=8 --n_updates=32 --gamma=0.999 --n_iters_train=2000 --n_iters_eval=64 --lr=0.001 --best_of_n_experts=10 --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets//real/minatar/name=Asterix-MinAtar;tl=500//"      
python icl_gen_ed.py --env_id="name=Breakout-MinAtar;tl=500"      --agent_id="minatar" --n_seeds_seq=1 --n_seeds_par=1 --n_envs=128 --n_envs_batch=8 --n_updates=32 --gamma=0.999 --n_iters_train=2000 --n_iters_eval=64 --lr=0.001 --best_of_n_experts=10 --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets//real/minatar/name=Breakout-MinAtar;tl=500//"     
python icl_gen_ed.py --env_id="name=Freeway-MinAtar;tl=500"       --agent_id="minatar" --n_seeds_seq=1 --n_seeds_par=1 --n_envs=128 --n_envs_batch=8 --n_updates=32 --gamma=0.999 --n_iters_train=2000 --n_iters_eval=64 --lr=0.001 --best_of_n_experts=10 --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets//real/minatar/name=Freeway-MinAtar;tl=500//"      
python icl_gen_ed.py --env_id="name=SpaceInvaders-MinAtar;tl=500" --agent_id="minatar" --n_seeds_seq=1 --n_seeds_par=1 --n_envs=128 --n_envs_batch=8 --n_updates=32 --gamma=0.999 --n_iters_train=2000 --n_iters_eval=64 --lr=0.001 --best_of_n_experts=10 --save_dir="/data/vision/phillipi/akumar01/synthetic-mdps-data/datasets//real/minatar/name=SpaceInvaders-MinAtar;tl=500//"