import experiment_utils
import run
import run_bc

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


def experiment_main(dir_exp, n_gpus):
    envs_train = [
        "name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=F;trans=linear;rew=linear;mrl=4x64",
        "name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=F;trans=linear;rew=goal;mrl=4x64",
        "name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=F;trans=mlp;rew=linear;mrl=4x64",
        # "name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=F;trans=mlp;rew=goal;mrl=4x64",
    ]
    envs_test = [
        "name=CartPole-v1;tl=500",
        "name=Acrobot-v1;tl=500",
        "name=MountainCar-v0;tl=500",
        "name=Asterix-MinAtar;tl=500",
        "name=Breakout-MinAtar;tl=500",
        # "name=Freeway-MinAtar;tl=500",
        "name=SpaceInvaders-MinAtar;tl=500",
    ]

    cfg_default = vars(run.parse_args())
    cfg_train = cfg_default.copy()
    cfg_train.update(n_seeds=1, agent_id="obs_embed=dense;name=linear_transformer;tl=500", run='train',
                     save_agent_params=True, n_envs=128, n_envs_batch=32, n_iters=4000)
    cfg_test = cfg_default.copy()
    cfg_test.update(n_seeds=1, agent_id="obs_embed=dense;name=linear_transformer;tl=500", run='train',
                    save_agent_params=True, n_envs=128, n_envs_batch=32, n_iters=4000, ft_first_last_layers=True, )

    cfgs = []
    for env_train in envs_train:
        cfg = cfg_train.copy()
        cfg["env_id"] = env_train
        cfg["load_dir"] = None
        cfg["save_dir"] = f"{dir_exp}/train/{env_train}"
        cfgs.append(cfg)
    txt_train = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python run.py')

    cfgs = []
    for env_test in envs_test:
        for env_train in envs_train:
            cfg = cfg_test.copy()
            cfg["env_id"] = env_test
            cfg["load_dir"] = f"{dir_exp}/train/{env_train}"
            cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{env_train}"
            cfgs.append(cfg)

        # eval random agent
        cfg = cfg_test.copy()
        cfg["env_id"] = env_test
        cfg["load_dir"] = None
        cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{'random_agent'}"
        cfg['run'] = 'eval'
        cfgs.append(cfg)

        # ft random agent
        cfg = cfg_test.copy()
        cfg["env_id"] = env_test
        cfg["load_dir"] = None
        cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{'ft_random'}"
        cfgs.append(cfg)

        # train random agent
        cfg = cfg_test.copy()
        cfg["env_id"] = env_test
        cfg["load_dir"] = None
        cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{'train_random'}"
        cfg["ft_first_last_layers"] = False
        cfgs.append(cfg)

    txt_eval = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python run.py')

    txt_train = change_to_n_gpus(txt_train, n_gpus)
    txt_eval = change_to_n_gpus(txt_eval, n_gpus)
    txt = f"{txt_header}\n{txt_train}\n{txt_eval}"
    return txt


def experiment_bc_transfer(dir_exp, n_gpus):
    envs_train = [
        "name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=F;trans=linear;rew=linear;mrl=4x64",
        "name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=F;trans=linear;rew=goal;mrl=4x64",
        "name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=F;trans=mlp;rew=linear;mrl=4x64",
        "name=csmdp;d_state=8;d_obs=8;n_acts=4;delta=F;trans=mlp;rew=goal;mrl=4x64",
    ]
    envs_test = [
        "name=CartPole-v1;tl=500",
        "name=Acrobot-v1;tl=500",
        "name=MountainCar-v0;tl=500",
        "name=Asterix-MinAtar;tl=500",
        "name=Breakout-MinAtar;tl=500",
        # "name=Freeway-MinAtar;tl=500",
        "name=SpaceInvaders-MinAtar;tl=500",
    ]

    ppo_cfg_default = vars(run.parse_args())
    bc_cfg_default = vars(run_bc.parse_args())
    print(ppo_cfg_default)
    print(bc_cfg_default)

    # ------------------- SYNTHETIC PRETRAINING: PPO -------------------
    cfgs = []
    cfg_shared = ppo_cfg_default.copy()
    cfg_shared.update(n_seeds=1, agent_id="obs_embed=dense;name=linear_transformer;tl=500", run='train',
                      save_agent_params=True, n_envs=128, n_envs_batch=32, n_iters=50)
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
    cfg_shared.update(n_seeds=1, agent_id="obs_embed=dense;name=linear_transformer;tl=500", run='train',
                      save_agent_params=True, n_envs=128, n_envs_batch=32, n_iters=50)
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
    cfg_shared.update(n_seeds=2, agent_id="obs_embed=dense;name=linear_transformer;tl=500", run='train',
                      save_agent_params=True, n_envs=4, n_envs_batch=4, n_iters=50, ft_first_last_layers=False, )
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
        cfg["run"] = 'eval'
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
    txt = f"{txt_header}\n{txt_pretrain}\n{txt_expert}\n{txt_eval}"
    return txt


if __name__ == '__main__':
    with open("experiment_tbc.sh", "w") as f:
        f.write(experiment_bc_transfer("../data/exp_tbc/", 6))
