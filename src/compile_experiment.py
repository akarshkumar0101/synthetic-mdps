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
    # envs_train = []
    # for n_acts in [2, 3, 5]:
    #     for tl in [4, 128]:
    #         denvu = f"env=dsmdp;n_states=64;n_acts={n_acts};d_obs=64;rdist=U;rpo=64;tl={tl}"
    #         denvn = f"env=dsmdp;n_states=64;n_acts={n_acts};d_obs=64;rdist=N;rpo=64;tl={tl}"
    #         cenv = f"env=csmdp;d_state=8;n_acts={n_acts};d_obs=64;delta=F;rpo=64;tl={tl}"
    #         cenvd = f"env=csmdp;d_state=8;n_acts={n_acts};d_obs=64;delta=T;rpo=64;tl={tl}"
    #         envs_train.extend([denvu, denvn, cenv, cenvd])
    #
    # envs_eval = [
    #     "env=gridenv;grid_len=8;fobs=T;rpo=64;tl=128",
    #     "env=cartpole;fobs=T;rpo=64;tl=128",
    #     "env=mountaincar;fobs=T;rpo=64;tl=128",
    #     "env=acrobot;fobs=T;rpo=64;tl=128",
    # ]

    envs_train = []
    for n_acts in [2, 3, 5]:
        for tl in [4, 128]:
            denvu = f"env=dsmdp;n_states=64;n_acts={n_acts};d_obs=64;rdist=U;rpo=64;tl={tl}"
            denvn = f"env=dsmdp;n_states=64;n_acts={n_acts};d_obs=64;rdist=N;rpo=64;tl={tl}"
            cenv = f"env=csmdp;d_state=8;n_acts={n_acts};d_obs=64;delta=F;rpo=64;tl={tl}"
            cenvd = f"env=csmdp;d_state=8;n_acts={n_acts};d_obs=64;delta=T;rpo=64;tl={tl}"
            envs_train.extend([denvu, denvn, cenv, cenvd])

    envs_eval = [
        "env=gridenv;grid_len=8;fobs=T;rpo=64;tl=128",
        "env=cartpole;fobs=T;rpo=64;tl=128",
        "env=mountaincar;fobs=T;rpo=64;tl=128",
        "env=acrobot;fobs=T;rpo=64;tl=128",
    ]
    config_default = dict(n_seeds=0, env_id=None, agent=None, run=None, load_dir=None, save_dir=None, n_iters=None)
    config_train = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="train", load_dir=None, save_dir=None,
                        n_iters=1000)
    config_eval = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="eval", load_dir=None, save_dir=None,
                       n_iters=10)

    configs = []
    for env_pre in envs_train:
        c = config_train.copy()
        c["env_id"] = env_pre
        c["save_dir"] = f"../data/{env_pre}"
        configs.append(c)
    txt_pre = experiment_utils.create_command_txt_from_configs(configs, config_default, python_command='python run.py')

    configs = []
    for env_pre in envs_train:
        for env_trans in envs_eval:
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


def experiment_mrl_horizon():
    envs_train = [
        f"name=dsmdp;n_states=64;n_acts={4};d_obs=16;rpo=16;mrl=1x128",
        f"name=dsmdp;n_states=64;n_acts={4};d_obs=16;rpo=16;mrl=8x16",
        f"name=dsmdp;n_states=64;n_acts={4};d_obs=16;rpo=16;mrl=16x8",
        f"name=dsmdp;n_states=64;n_acts={4};d_obs=16;rpo=16;mrl=128x1",
    ]
    envs_eval = envs_train

    config_default = dict(n_seeds=0, env_id=None, agent=None, run=None, load_dir=None, save_dir=None, n_iters=None)
    config_train = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="train", load_dir=None, save_dir=None,
                        n_iters=3000, n_envs=128, n_envs_batch=32, lr=1e-4)
    config_eval = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="eval", load_dir=None, save_dir=None,
                       n_iters=10, n_envs=128, n_envs_batch=32, lr=1e-4)

    configs = []
    for env_train in envs_train:
        c = config_train.copy()
        c["env_id"] = env_train
        c["save_dir"] = f"../data/train/{env_train}"
        configs.append(c)
    txt_train = experiment_utils.create_command_txt_from_configs(configs, config_default,
                                                                 python_command='python run.py')

    configs = []
    for env_train in [*envs_train, None]:
        for env_eval in envs_eval:
            if env_train is not None and get_n_acts(env_train) != get_n_acts(env_eval):
                continue
            c = config_eval.copy()
            c["env_id"] = env_eval
            c["load_dir"] = f"../data/train/{env_train}" if env_train is not None else None
            c["save_dir"] = f"../data/eval/{env_eval}/{env_train}"
            configs.append(c)
    txt_eval = experiment_utils.create_command_txt_from_configs(configs, config_default, python_command='python run.py')

    txt_train = change_to_n_gpus(txt_train, 4)
    txt_eval = change_to_n_gpus(txt_eval, 4)

    txt = "\n".join([txt_header, txt_train, txt_eval])

    with open("experiment.sh", "w") as f:
        f.write(txt)


def experiment_transfer():
    envs_train = [
        "name=gridenv;grid_len=8;pos_start=random;pos_rew=random;fobs=T;rpo=16;mrl=4x32",
        "name=cartpole;fobs=T;rpo=16;mrl=4x32",
        "name=mountaincar;fobs=T;rpo=16;mrl=4x32",
        "name=acrobot;fobs=T;rpo=16;mrl=4x32",
    ]
    for n_acts in [2, 3, 5]:
        denv = f"name=dsmdp;n_states=64;n_acts={n_acts};d_obs=16;rpo=16;mrl=4x32"
        # cenv = f"name=csmdp;d_state=4;n_acts={n_acts};d_obs=16;delta=F;rpo=16;mrl=4x32"
        cenvd = f"name=csmdp;d_state=4;n_acts={n_acts};d_obs=16;delta=T;rpo=16;mrl=4x32"
        envs_train.extend([denv, cenvd])
    envs_eval = envs_train

    config_default = dict(n_seeds=0, env_id=None, agent=None, run=None, load_dir=None, save_dir=None, n_iters=None)
    config_train = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="train", load_dir=None, save_dir=None,
                        n_iters=5000, n_envs=128, n_envs_batch=32, lr=1e-4)
    config_eval = dict(n_seeds=8, env_id=None, agent="linear_transformer", run="eval", load_dir=None, save_dir=None,
                       n_iters=10, n_envs=128, n_envs_batch=32, lr=1e-4)

    configs = []
    for env_train in envs_train:
        c = config_train.copy()
        c["env_id"] = env_train
        c["save_dir"] = f"../data/train/{env_train}"
        configs.append(c)
    txt_train = experiment_utils.create_command_txt_from_configs(configs, config_default,
                                                                 python_command='python run.py')

    configs = []
    for env_train in [*envs_train, None]:
        for env_eval in envs_eval:
            if env_train is not None and get_n_acts(env_train) != get_n_acts(env_eval):
                continue
            c = config_eval.copy()
            c["env_id"] = env_eval
            c["load_dir"] = f"../data/train/{env_train}" if env_train is not None else None
            c["save_dir"] = f"../data/eval/{env_eval}/{env_train}"
            configs.append(c)
    txt_eval = experiment_utils.create_command_txt_from_configs(configs, config_default, python_command='python run.py')

    txt_train = change_to_n_gpus(txt_train, 4)
    txt_eval = change_to_n_gpus(txt_eval, 4)

    txt = "\n".join([txt_header, txt_train, txt_eval])

    with open("experiment.sh", "w") as f:
        f.write(txt)


if __name__ == '__main__':
    experiment_transfer()
