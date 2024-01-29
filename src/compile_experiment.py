import experiment_utils
import icl_bc
import icl_gen

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


def get_n_acts(env):
    config = dict([sub.split('=') for sub in env.split(';')])
    try:
        return int(config["n_acts"])
    except KeyError:
        return dict(gridenv=5, cartpole=2, mountaincar=3, acrobot=3)[config["name"]]


def change_to_n_gpus(txt, n_gpus):
    lines = [line for line in txt.split("\n") if line]
    out = []
    for i, line in enumerate(lines):
        out.append(f'CUDA_VISIBLE_DEVICES={i % n_gpus} {line} &')
        if i % n_gpus == n_gpus - 1 or i == len(lines) - 1:
            out.append("wait")
    out = "\n".join(out) + "\n"
    return out


txt_header = "\n".join(["#!/bin/bash",
                        "source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate",
                        "cd /data/vision/phillipi/akumar01/synthetic-mdps/src", "\n" * 3])


def exp_iclbc(dir_exp, n_gpus):
    # envs_train = [
    #     "name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64",
    #     "name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64",
    #     # "name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=linear;mrl=4x64",
    #     # "name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=goal;mrl=4x64",
    #     # "name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64",
    #     # "name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64",
    # ]
    # envs_test = [
    #     "name=CartPole-v1;tl=256",
    #     # "name=CartPole-v1;tl=500",
    #     # "name=Acrobot-v1;tl=500",
    #     # "name=MountainCar-v0;tl=500",
    #     # "name=Asterix-MinAtar;tl=500",
    #     # "name=Breakout-MinAtar;tl=500",
    #     # "name=Freeway-MinAtar;tl=500",
    #     # "name=SpaceInvaders-MinAtar;tl=500",
    # ]
    envs_train = [
        "name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=goal;tl=64",
        "name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=linear;tl=64",
    ]
    envs_test = [
        "name=CartPole-v1",
        "name=Acrobot-v1",
        "name=MountainCar-v0",
        "name=DiscretePendulum-v1",
        "name=Asterix-MinAtar",
        "name=Breakout-MinAtar",
        "name=Freeway-MinAtar",
        "name=SpaceInvaders-MinAtar",
    ]

    cfg_default = vars(icl_bc.parse_args())
    print(cfg_default)

    # ------------------- VIZ envs -------------------
    # cfgs = []
    # for env_train in envs_train:
    #     cfg = viz_cfg_default.copy()
    #     cfg.update(env_id=env_train, save_dir=f"{dir_exp}/viz/{env_train}")
    #     cfgs.append(cfg)
    # txt_viz = experiment_utils.create_command_txt_from_configs(cfgs, viz_cfg_default,
    #                                                            python_command='python viz_util.py')

    # ------------------- TRAIN -------------------
    cfgs = []
    for env_id_train in envs_train:
        cfg = cfg_default.copy()
        cfg.update(
            name=f"pretrain_{env_id_train}",
            n_iters=2000,
            dataset_path=f'{dir_exp}/datasets/{env_id_train}/dataset.pkl',
            save_dir=f"{dir_exp}/train/{env_id_train}",
            time_perm=False,
            save_agent=True,
        )
        cfgs.append(cfg)
    txt_train = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_bc.py')

    # ------------------- TEST -------------------
    cfgs = []
    for env_id_test in envs_test:
        for env_id_train in [*envs_train, None]:
            cfg = cfg_default.copy()
            cfg.update(
                name=f"train_{env_id_train}_test_{env_id_test}",
                n_iters=1000,
                dataset_path=f'{dir_exp}/datasets/{env_id_test}/dataset.pkl',
                load_dir=f"{dir_exp}/train/{env_id_train}" if env_id_train is not None else None,
                save_dir=f"{dir_exp}/test/{env_id_test}/{env_id_train}",
                time_perm=False
            )
            if env_id_train is not None:
                cfg.update()

            cfgs.append(cfg)
    txt_test = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_bc.py')

    txt_train = change_to_n_gpus(txt_train, n_gpus)
    txt_test = change_to_n_gpus(txt_test, n_gpus)
    txt = f"{txt_header}\n\n{txt_train}\n{txt_test}"
    return txt


def exp_generate_data(dir_exp, n_gpus):
    cfg_default = vars(icl_gen.parse_args())
    print(cfg_default)

    envs_synthetic = [
        "name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=goal;tl=64",
        "name=csmdp;d_state=2;d_obs=4;n_acts=4;delta=T;trans=linear;rew=linear;tl=64",
    ]

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
            n_iters_train=2000,
            n_iters_eval=300,
            lr=3e-4,
            best_of_n_experts=10,
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

    txt_syn = change_to_n_gpus(txt_syn, n_gpus)
    txt_cla = change_to_n_gpus(txt_cla, n_gpus)
    txt_ma = change_to_n_gpus(txt_ma, n_gpus)

    txt = f"{txt_header}\n\n{txt_syn}\n{txt_cla}\n{txt_ma}"
    return txt


if __name__ == '__main__':
    # with open("experiment.sh", "w") as f:
    #     f.write(exp_generate_data("../data/exp_iclbc/", 6))

    with open("experiment.sh", "w") as f:
        f.write(exp_iclbc("../data/exp_iclbc/", 6))
