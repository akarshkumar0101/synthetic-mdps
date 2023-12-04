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

import experiment_utils


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


def main():
    envs_pre = []
    # for tl in [1, 128]:
    #     env = f"name=dsmdp;n_states=64;n_acts={4};d_obs=64;rdist=N;rpo=64;tl={tl}"
    #     envs_pre.append(env)
    envs_pre = ["name=gridenv;grid_len=8;fobs=T;rpo=64;tl=128"]
    envs_transfer = envs_pre

    # envs_pre = []
    # for n_acts in [2, 3, 4]:
    #     for tl in [4, 128]:
    #         denvu = f"env=dsmdp;n_states=64;n_acts={n_acts};d_obs=64;rdist=U;rpo=64;tl={tl}"
    #         denvn = f"env=dsmdp;n_states=64;n_acts={n_acts};d_obs=64;rdist=N;rpo=64;tl={tl}"
    #         cenv = f"env=csmdp;d_state=8;n_acts={n_acts};d_obs=64;delta=F;rpo=64;tl={tl}"
    #         cenvd = f"env=csmdp;d_state=8;n_acts={n_acts};d_obs=64;delta=T;rpo=64;tl={tl}"
    #         envs_pre.extend([denvu, denvn, cenv, cenvd])
    #
    # envs_transfer = [
    #     "env=gridenv;grid_len=8;fobs=T;rpo=64;tl=128",
    #     "env=cartpole;fobs=T;rpo=64;tl=128",
    #     "env=mountaincar;fobs=T;rpo=64;tl=128",
    #     "env=acrobot;fobs=T;rpo=64;tl=128",
    # ]
    config_default = dict(n_seeds=0, env_id=None, agent=None, run=None, load_dir=None, save_dir=None, n_iters=None)
    config_train = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="train", load_dir=None, save_dir=None,
                        n_iters=1000)
    config_eval = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="eval", load_dir=None, save_dir=None,
                       n_iters=10)

    configs = []
    for env_pre in envs_pre:
        c = config_train.copy()
        c["env_id"] = env_pre
        c["save_dir"] = f"../data/{env_pre}"
        configs.append(c)
    txt_pre = experiment_utils.create_command_txt_from_configs(configs, config_default, python_command='python run.py')

    configs = []
    for env_pre in envs_pre:
        for env_trans in envs_transfer:
            if get_n_acts(env_pre) != get_n_acts(env_trans):
                continue
            c = config_eval.copy()
            c["env_id"] = env_trans
            c["save_dir"] = f"../data/transfer/{env_trans}/{env_pre}"
            c["load_dir"] = f"../data/{env_pre}"
            configs.append(c)
    txt_trans = experiment_utils.create_command_txt_from_configs(configs, config_default,
                                                                 python_command='python run.py')

    txt_pre = change_to_n_gpus(txt_pre, 4)
    txt_trans = change_to_n_gpus(txt_trans, 4)
    txt = txt_pre + "\n" * 5 + txt_trans

    txt = "\n".join(["#!/bin/bash",
                     "source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate",
                     "cd /data/vision/phillipi/akumar01/synthetic-mdps/src", "\n" * 3]) + txt

    with open("experiment.sh", "w") as f:
        f.write(txt)


def some_experiment():
    envs = ["name=cartpole;fobs=T;tl=128", "name=mountaincar;fobs=T;tl=128", "name=acrobot;fobs=T;tl=128", ]

    config_default = dict(n_seeds=0, env_id=None, agent=None, run=None, load_dir=None, save_dir=None, n_iters=None)
    config_train = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="train", load_dir=None, save_dir=None,
                        n_iters=1000)
    config_eval = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="eval", load_dir=None, save_dir=None,
                       n_iters=10)

    configs = []
    for env in envs:
        c = config_train.copy()
        c["env_id"] = env
        c["save_dir"] = f"../data/train/{env}"
        configs.append(c)
    txt_train = experiment_utils.create_command_txt_from_configs(configs, config_default,
                                                                 python_command='python run.py')

    configs = []
    for env in envs:
        c = config_eval.copy()
        c["env_id"] = env
        c["load_dir"] = f"../data/train/{env}"
        c["save_dir"] = f"../data/eval/{env}/{env}"
        configs.append(c)
    txt_eval = experiment_utils.create_command_txt_from_configs(configs, config_default, python_command='python run.py')

    txt_train = change_to_n_gpus(txt_train, 4)
    txt_eval = change_to_n_gpus(txt_eval, 4)

    txt = "\n".join([txt_header, txt_train, txt_eval])

    with open("experiment.sh", "w") as f:
        f.write(txt)


if __name__ == '__main__':
    some_experiment()