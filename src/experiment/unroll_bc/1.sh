#!/bin/bash
source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate
cd /data/vision/phillipi/akumar01/synthetic-mdps/src



CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64/ckpt_0.pkl"                                       &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64/ckpt_0.pkl"                                       &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/name=rf;t_a=3;t_c=4;o_d=0/ckpt_0.pkl"                                                            &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/name=rf;t_a=1;t_c=3;o_d=0/ckpt_0.pkl"                                                            &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/name=rf;t_a=0;t_c=1;o_d=4/ckpt_2.pkl"                                                            &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/zero_act/ckpt_2.pkl"                                                                             &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/name=CartPole-v1/ckpt_2.pkl"                                                                     &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/all/ckpt_2.pkl"                                                                                  &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/n-1/ckpt_4.pkl"                                                                                  &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/scratch/ckpt_4.pkl"                                                                              &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64/ckpt_final.pkl"           &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=CartPole-v1//dataset.pkl"           --env_id="name=CartPole-v1"           --ckpt_path="../data/exp_icl//test_bc/name=CartPole-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64/ckpt_final.pkl"           &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64/ckpt_1.pkl"                &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64/ckpt_1.pkl"                                        &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64/ckpt_1.pkl"                                        &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64/ckpt_1.pkl"                                        &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=3;t_c=4;o_d=0/ckpt_3.pkl"                                                             &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=1;t_c=3;o_d=0/ckpt_3.pkl"                                                             &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/name=rf;t_a=0;t_c=1;o_d=4/ckpt_3.pkl"                                                             &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/zero_act/ckpt_3.pkl"                                                                              &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/name=Acrobot-v1/ckpt_final.pkl"                                                                   &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/all/ckpt_final.pkl"                                                                               &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/n-1/ckpt_final.pkl"                                                                               &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=Acrobot-v1//dataset.pkl"            --env_id="name=Acrobot-v1"            --ckpt_path="../data/exp_icl//test_bc/name=Acrobot-v1/scratch/ckpt_final.pkl"                                                                           &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --env_id="name=MountainCar-v0"        --ckpt_path="../data/exp_icl//test_bc/name=MountainCar-v0/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64/ckpt_2.pkl"            &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --env_id="name=MountainCar-v0"        --ckpt_path="../data/exp_icl//test_bc/name=MountainCar-v0/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64/ckpt_2.pkl"            &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --env_id="name=MountainCar-v0"        --ckpt_path="../data/exp_icl//test_bc/name=MountainCar-v0/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64/ckpt_2.pkl"            &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --env_id="name=MountainCar-v0"        --ckpt_path="../data/exp_icl//test_bc/name=MountainCar-v0/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64/ckpt_2.pkl"                                    &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --env_id="name=MountainCar-v0"        --ckpt_path="../data/exp_icl//test_bc/name=MountainCar-v0/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64/ckpt_4.pkl"                                    &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --env_id="name=MountainCar-v0"        --ckpt_path="../data/exp_icl//test_bc/name=MountainCar-v0/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64/ckpt_4.pkl"                                    &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --env_id="name=MountainCar-v0"        --ckpt_path="../data/exp_icl//test_bc/name=MountainCar-v0/name=rf;t_a=3;t_c=4;o_d=0/ckpt_4.pkl"                                                         &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=MountainCar-v0//dataset.pkl"        --env_id="name=MountainCar-v0"        --ckpt_path="../data/exp_icl//test_bc/name=MountainCar-v0/name=rf;t_a=1;t_c=3;o_d=0/ckpt_4.pkl"                                                         &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=rf;t_a=0;t_c=1;o_d=4/ckpt_0.pkl"                                                    &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/zero_act/ckpt_0.pkl"                                                                     &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=DiscretePendulum-v1/ckpt_0.pkl"                                                     &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/all/ckpt_0.pkl"                                                                          &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/n-1/ckpt_2.pkl"                                                                          &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/scratch/ckpt_2.pkl"                                                                      &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64/ckpt_3.pkl"       &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64/ckpt_3.pkl"       &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64/ckpt_final.pkl"   &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64/ckpt_final.pkl"                           &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64/ckpt_final.pkl"                           &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/classic/name=DiscretePendulum-v1//dataset.pkl"   --env_id="name=DiscretePendulum-v1"   --ckpt_path="../data/exp_icl//test_bc/name=DiscretePendulum-v1/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64/ckpt_final.pkl"                           &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --env_id="name=Asterix-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=rf;t_a=3;t_c=4;o_d=0/ckpt_1.pkl"                                                        &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --env_id="name=Asterix-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=rf;t_a=1;t_c=3;o_d=0/ckpt_1.pkl"                                                        &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --env_id="name=Asterix-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=rf;t_a=0;t_c=1;o_d=4/ckpt_1.pkl"                                                        &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --env_id="name=Asterix-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Asterix-MinAtar/zero_act/ckpt_1.pkl"                                                                         &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --env_id="name=Asterix-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Asterix-MinAtar/name=Asterix-MinAtar/ckpt_3.pkl"                                                             &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --env_id="name=Asterix-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Asterix-MinAtar/all/ckpt_3.pkl"                                                                              &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --env_id="name=Asterix-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Asterix-MinAtar/n-1/ckpt_3.pkl"                                                                              &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Asterix-MinAtar//dataset.pkl"       --env_id="name=Asterix-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Asterix-MinAtar/scratch/ckpt_3.pkl"                                                                          &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64/ckpt_0.pkl"          &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64/ckpt_0.pkl"          &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64/ckpt_0.pkl"          &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64/ckpt_0.pkl"                                  &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64/ckpt_2.pkl"                                  &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64/ckpt_2.pkl"                                  &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=3;t_c=4;o_d=0/ckpt_2.pkl"                                                       &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=1;t_c=3;o_d=0/ckpt_2.pkl"                                                       &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=rf;t_a=0;t_c=1;o_d=4/ckpt_4.pkl"                                                       &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/zero_act/ckpt_4.pkl"                                                                        &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/name=Breakout-MinAtar/ckpt_4.pkl"                                                           &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Breakout-MinAtar//dataset.pkl"      --env_id="name=Breakout-MinAtar"      --ckpt_path="../data/exp_icl//test_bc/name=Breakout-MinAtar/all/ckpt_4.pkl"                                                                             &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/n-1/ckpt_0.pkl"                                                                              &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/scratch/ckpt_0.pkl"                                                                          &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64/ckpt_1.pkl"           &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64/ckpt_1.pkl"           &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64/ckpt_3.pkl"           &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64/ckpt_3.pkl"                                   &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=dsmdp;i_d=1;i_s=1;t_a=1;t_s=0;o_d=2;tl=64/ckpt_3.pkl"                                   &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=dsmdp;i_d=4;i_s=3;t_a=3;t_s=2;o_d=4;tl=64/ckpt_3.pkl"                                   &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=rf;t_a=3;t_c=4;o_d=0/ckpt_final.pkl"                                                    &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=rf;t_a=1;t_c=3;o_d=0/ckpt_final.pkl"                                                    &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/name=rf;t_a=0;t_c=1;o_d=4/ckpt_final.pkl"                                                    &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=Freeway-MinAtar//dataset.pkl"       --env_id="name=Freeway-MinAtar"       --ckpt_path="../data/exp_icl//test_bc/name=Freeway-MinAtar/zero_act/ckpt_final.pkl"                                                                     &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" --env_id="name=SpaceInvaders-MinAtar" --ckpt_path="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=SpaceInvaders-MinAtar/ckpt_1.pkl"                                                 &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" --env_id="name=SpaceInvaders-MinAtar" --ckpt_path="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/all/ckpt_1.pkl"                                                                        &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" --env_id="name=SpaceInvaders-MinAtar" --ckpt_path="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/n-1/ckpt_1.pkl"                                                                        &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" --env_id="name=SpaceInvaders-MinAtar" --ckpt_path="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/scratch/ckpt_1.pkl"                                                                    &
wait
CUDA_VISIBLE_DEVICES=0 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" --env_id="name=SpaceInvaders-MinAtar" --ckpt_path="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=csmdp;i_d=4;i_s=0;t_a=3;t_c=3;t_l=3;t_s=1;o_d=3;o_c=2;r_c=4;tl=64/ckpt_4.pkl"     &
CUDA_VISIBLE_DEVICES=1 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" --env_id="name=SpaceInvaders-MinAtar" --ckpt_path="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=csmdp;i_d=0;i_s=0;t_a=4;t_c=2;t_l=1;t_s=0;o_d=1;o_c=1;r_c=0;tl=64/ckpt_4.pkl"     &
CUDA_VISIBLE_DEVICES=2 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" --env_id="name=SpaceInvaders-MinAtar" --ckpt_path="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=csmdp;i_d=1;i_s=4;t_a=3;t_c=0;t_l=3;t_s=0;o_d=2;o_c=3;r_c=0;tl=64/ckpt_4.pkl"     &
CUDA_VISIBLE_DEVICES=3 python unroll.py --dataset_path="../data/exp_icl//datasets//real/minatar/name=SpaceInvaders-MinAtar//dataset.pkl" --env_id="name=SpaceInvaders-MinAtar" --ckpt_path="../data/exp_icl//test_bc/name=SpaceInvaders-MinAtar/name=dsmdp;i_d=1;i_s=3;t_a=3;t_s=3;o_d=0;tl=64/ckpt_4.pkl"                             &
wait
