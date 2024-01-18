import experiment_utils
import run
import run_bc
import viz_util

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


def experiment(dir_exp, n_gpus):
    agent_id = "obs_embed=dense;name=linear_transformer;tl=256"
    envs_train = [
        "name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64",
        "name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64",
        "name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=linear;mrl=4x64",
        "name=csmdp;d_state=2;d_obs=4;n_acts=2;delta=T;trans=mlp;rew=goal;mrl=4x64",
        "name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=linear;mrl=4x64",
        "name=csmdp;d_state=4;d_obs=4;n_acts=2;delta=T;trans=linear;rew=goal;mrl=4x64",
    ]
    envs_test = [
        "name=CartPole-v1;tl=256",
        # "name=CartPole-v1;tl=500",
        # "name=Acrobot-v1;tl=500",
        # "name=MountainCar-v0;tl=500",
        # "name=Asterix-MinAtar;tl=500",
        # "name=Breakout-MinAtar;tl=500",
        # "name=Freeway-MinAtar;tl=500",
        # "name=SpaceInvaders-MinAtar;tl=500",
    ]

    viz_cfg_default = vars(viz_util.parser.parse_args())
    ppo_cfg_default = vars(run.parse_args())
    bc_cfg_default = vars(run_bc.parse_args())
    print(ppo_cfg_default)
    print(bc_cfg_default)

    # ------------------- VIZ envs -------------------
    cfgs = []
    for env_train in envs_train:
        cfg = viz_cfg_default.copy()
        cfg.update(env_id=env_train, save_dir=f"{dir_exp}/viz/{env_train}")
        cfgs.append(cfg)
    txt_viz = experiment_utils.create_command_txt_from_configs(cfgs, viz_cfg_default,
                                                               python_command='python viz_util.py')

    # ------------------- SYNTHETIC PRETRAINING: PPO -------------------
    cfgs = []
    cfg_shared = ppo_cfg_default.copy()
    cfg_shared.update(n_seeds=1, agent_id=agent_id, run='train',
                      save_agent_params=True, n_envs=128, n_envs_batch=32, n_iters=10000)  # 10000
    for env_train in envs_train:
        cfg = cfg_shared.copy()

        cfg["env_id"] = env_train
        cfg["load_dir"] = None
        cfg["save_dir"] = f"{dir_exp}/pretrain/{env_train}"
        cfgs.append(cfg)
    txt_pretrain = experiment_utils.create_command_txt_from_configs(cfgs, ppo_cfg_default,
                                                                    python_command='python run.py')

    # ------------------- EXPERT TRAINING: PPO -------------------
    cfgs = []
    cfg_shared = ppo_cfg_default.copy()
    cfg_shared.update(n_seeds=4, agent_id=agent_id, run='train',
                      save_agent_params=True, n_envs=128, n_envs_batch=32, n_iters=1000)  # 1000
    for env_test in envs_test:
        cfg = cfg_shared.copy()

        cfg["env_id"] = env_test
        cfg["load_dir"] = None
        cfg["save_dir"] = f"{dir_exp}/expert/{env_test}"
        cfgs.append(cfg)
    txt_expert = experiment_utils.create_command_txt_from_configs(cfgs, ppo_cfg_default, python_command='python run.py')

    # ------------------- FINE-TUNE EVALUATION: ICL -------------------
    # actually, this doesn't work yet because new envs have different obs shape and act space

    # ------------------- FINE-TUNE EVALUATION: BC -------------------
    cfgs = []
    cfg_shared = bc_cfg_default.copy()
    cfg_shared.update(n_seeds=32, agent_id=agent_id, run='train',
                      save_agent_params=False, n_envs=4, n_envs_batch=4, n_iters=300,  # 300
                      reset_layers='last', ft_layers='last')
    for env_test in envs_test:
        for env_train in envs_train:
            cfg = cfg_shared.copy()
            cfg["env_id"] = env_test
            cfg["load_dir"] = f"{dir_exp}/pretrain/{env_train}"
            cfg["load_dir_teacher"] = f"{dir_exp}/expert/{env_test}"
            cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{env_train}"
            cfgs.append(cfg)

        # random agent - no train
        cfg = cfg_shared.copy()
        cfg["env_id"] = env_test
        cfg["load_dir"] = None
        cfg["load_dir_teacher"] = f"{dir_exp}/expert/{env_test}"
        cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{'random_agent'}"
        cfg["lr"] = 0.0
        cfgs.append(cfg)

        # random agent - train
        cfg = cfg_shared.copy()
        cfg["env_id"] = env_test
        cfg["load_dir"] = None
        cfg["load_dir_teacher"] = f"{dir_exp}/expert/{env_test}"
        cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{'train_random'}"
        cfgs.append(cfg)
    txt_eval = experiment_utils.create_command_txt_from_configs(cfgs, bc_cfg_default, python_command='python run_bc.py')

    txt_pretrain = change_to_n_gpus(txt_pretrain, n_gpus)
    txt_expert = change_to_n_gpus(txt_expert, n_gpus)
    txt_eval = change_to_n_gpus(txt_eval, n_gpus)
    # txt = f"{txt_header}\n{txt_viz}\n{txt_pretrain}\n{txt_expert}\n{txt_eval}"
    txt = f"{txt_header}\n\n{txt_pretrain}\n{txt_expert}\n{txt_eval}"
    return txt


if __name__ == '__main__':
    with open("experiment.sh", "w") as f:
        f.write(experiment("../data/exp_01_17/", 6))
