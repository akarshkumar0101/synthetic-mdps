python icl_gen.py --env_id="name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64" --save_dir="../data/exp_icl//datasets//synthetic/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64//" --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003
python icl_gen.py --env_id="name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64" --save_dir="../data/exp_icl//datasets//synthetic/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64//" --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003
python icl_gen.py --env_id="random_function"                                                        --save_dir="../data/exp_icl//datasets//synthetic/random_function//"                                                        --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003
python icl_gen.py --env_id="zero_act"                                                               --save_dir="../data/exp_icl//datasets//synthetic/zero_act//"                                                               --n_seeds_seq=16 --n_seeds_par=16 --n_iters_train=100 --lr=0.0003
