import math
import os

import numpy as np

import experiment_utils
import icl_bc
import icl_gen

txt_header = "\n".join(["#!/bin/bash",
                        "source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate",
                        "cd /data/vision/phillipi/akumar01/synthetic-mdps/src", "\n" * 3])

"""
Pretraining Tasks:
"name=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=4"
"name=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=128"

"name=csmdp;d_state=8;n_acts=4;d_obs=64;delta=F;rpo=64;tl=128"
"name=csmdp;d_state=8;n_acts=4;d_obs=64;delta=T;rpo=64;tl=128"

Transfer tasks:
"name=gridenv;grid_len=8;fobs=T;rpo=64;tl=128"
"name=cartpole;fobs=T;rpo=64;tl=128"
"name=mountaincar;fobs=T;rpo=64;tl=128"
"name=acrobot;fobs=T;rpo=64;tl=128"
"""

# def get_n_acts(env):
#     config = dict([sub.split('=') for sub in env.split(';')])
#     try:
#         return int(config["n_acts"])
#     except KeyError:
#         return dict(gridenv=5, cartpole=2, mountaincar=3, acrobot=3)[config["name"]]

np.random.seed(0)
envs_synthetic = []
for i in range(128):
    i_d, i_s, t_a, t_c, t_l, t_s, o_d, o_c, r_c = [np.random.randint(0, 5) for _ in range(9)]
    env_id = f"name=csmdp;i_d={i_d};i_s={i_s};t_a={t_a};t_c={t_c};t_l={t_l};t_s={t_s};o_d={o_d};o_c={o_c};r_c={r_c};tl=64"
    envs_synthetic.append(env_id)

envs_classic = [
    "name=CartPole-v1",
    "name=Acrobot-v1",
    "name=MountainCar-v0",
    "name=DiscretePendulum-v1",
]

envs_minatar = [
    "name=Asterix-MinAtar",
    "name=Breakout-MinAtar",
    "name=Freeway-MinAtar",
    "name=SpaceInvaders-MinAtar",
]

envs_train = envs_synthetic
envs_test = envs_classic + envs_minatar


# def change_to_n_gpus(txt, n_gpus):
#     lines = [line for line in txt.split("\n") if line]
#     out = []
#     for i, line in enumerate(lines):
#         out.append(f'CUDA_VISIBLE_DEVICES={i % n_gpus} {line} &')
#         if i % n_gpus == n_gpus - 1 or i == len(lines) - 1:
#             out.append("wait")
#     out = "\n".join(out) + "\n"
#     return out


def exp_train(dir_exp, obj="bc"):
    cfg_default = vars(icl_bc.parse_args())
    # print(cfg_default)

    cfgs = []
    for env_id_train in envs_train:
        cfg = cfg_default.copy()
        cfg.update(
            name=f"pretrain_{env_id_train}",
            n_iters=2000,
            dataset_path=f'{dir_exp}/datasets/{env_id_train}/dataset.pkl',
            save_dir=f"{dir_exp}/train_{obj}/{env_id_train}",
            time_perm=False,
            obj=obj,
            save_agent=True,
        )
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_bc.py')
    return txt


def exp_test(dir_exp, obj="bc"):
    cfg_default = vars(icl_bc.parse_args())
    # print(cfg_default)

    cfgs = []
    for env_id_test in envs_test:
        for env_id_train in [*envs_train, None]:
            cfg = cfg_default.copy()
            cfg.update(
                name=f"train_{env_id_train}_test_{env_id_test}",
                n_iters=100,
                dataset_path=f'{dir_exp}/datasets/{env_id_test}/dataset.pkl',
                load_dir=f"{dir_exp}/train_{obj}/{env_id_train}" if env_id_train is not None else None,
                save_dir=f"{dir_exp}/test_{obj}/{env_id_test}/{env_id_train}",
                time_perm=False,
                obj=obj,
            )
            if env_id_train is not None:
                cfg.update()

            cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_bc.py')
    return txt


def exp_data(dir_exp):
    cfg_default = vars(icl_gen.parse_args())
    print(cfg_default)

    # ------------------- SYNTHETIC -------------------
    cfgs = []
    for env_id in envs_synthetic:
        cfg = cfg_default.copy()
        cfg.update(
            env_id=env_id,
            agent_id="small",
            n_seeds_seq=16,
            n_seeds_par=16,
            n_iters_train=100,
            n_iters_eval=1,
            lr=3e-4,
            save_dir=f"{dir_exp}/datasets/{env_id}/",
        )
        cfgs.append(cfg)
    txt_syn = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_gen.py')

    # ------------------- CLASSIC -------------------
    cfgs = []
    for env_id in envs_classic:
        cfg = cfg_default.copy()
        cfg.update(
            env_id=env_id,
            agent_id="classic",
            n_seeds_seq=1,
            n_seeds_par=1,
            n_iters_train=4000,
            n_iters_eval=300,
            lr=3e-4,
            best_of_n_experts=20,
            save_dir=f"{dir_exp}/datasets/{env_id}/",
        )
        cfgs.append(cfg)
    txt_cla = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_gen.py')

    # ------------------- MINATAR -------------------
    cfgs = []
    for env_id in envs_minatar:
        cfg = cfg_default.copy()
        cfg.update(
            env_id=env_id,
            agent_id="minatar",
            n_seeds_seq=1,
            n_seeds_par=1,
            n_envs=64,
            n_envs_batch=8,
            n_updates=32,
            gamma=.999,
            n_iters_train=2000,
            n_iters_eval=20,
            lr=1e-3,
            best_of_n_experts=10,
            save_dir=f"{dir_exp}/datasets/{env_id}/",
        )
        cfgs.append(cfg)
    txt_ma = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_gen.py')
    return f"{txt_syn}\n{txt_cla}\n{txt_ma}"


def write_to_nodes_gpus(d, txt, n_nodes=1, n_gpus=1):
    assert n_nodes > 0 and n_gpus > 0
    lines = [line for line in txt.split("\n") if line]

    txt_nodes = [txt_header for _ in range(n_nodes)]
    for i in range(int(math.ceil(len(lines) / n_gpus))):
        block = lines[i * n_gpus:(i + 1) * n_gpus]
        block = [f'CUDA_VISIBLE_DEVICES={i} {line} &' for i, line in enumerate(block)]
        block.append('wait')
        block = "\n".join(block) + "\n"

        i_node = i % n_nodes
        txt_nodes[i_node] += block

    for i_node, txt_node in enumerate(txt_nodes):
        os.makedirs(f"{d}", exist_ok=True)
        with open(f"{d}/{i_node}.sh", "w") as f:
            f.write(txt_node)


if __name__ == '__main__':
    dir_exp = "../data/exp_icl/"
    n_nodes, n_gpus = 4, 4
    os.system("rm -rf ./experiment/")
    os.makedirs("./experiment/", exist_ok=True)

    txt = exp_data(dir_exp)
    write_to_nodes_gpus("./experiment/data/", txt, n_nodes, n_gpus)

    txt = exp_train(dir_exp, obj="bc")
    write_to_nodes_gpus("./experiment/train_bc/", txt, n_nodes, n_gpus)

    txt = exp_test(dir_exp, obj="bc")
    write_to_nodes_gpus("./experiment/test_bc/", txt, n_nodes, n_gpus)

    txt = exp_train(dir_exp, obj="wm")
    write_to_nodes_gpus("./experiment/train_wm/", txt, n_nodes, n_gpus)

    txt = exp_test(dir_exp, obj="wm")
    write_to_nodes_gpus("./experiment/test_wm/", txt, n_nodes, n_gpus)
