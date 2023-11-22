"""
Pretraining Tasks:
"env=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=4"
"env=dsmdp;n_states=64;n_acts=4;d_obs=64;rpo=64;tl=128"

"env=csmdp;d_state=8;n_acts=4;d_obs=64;delta=F;rpo=64;tl=128"
"env=csmdp;d_state=8;n_acts=4;d_obs=64;delta=T;rpo=64;tl=128"

Transfer tasks:
"env=gridenv;grid_len=8;fobs=T;rpo=64;tl=128"
"env=cartpole;fobs=T;rpo=64;tl=128"
"env=mountaincar;fobs=T;rpo=64;tl=128"
"env=acrobot;fobs=T;rpo=64;tl=128"
"""

import experiment_utils


def get_n_acts(env):
    config = dict([sub.split('=') for sub in env.split(';')])
    try:
        return int(config["n_acts"])
    except KeyError:
        return dict(gridenv=4, cartpole=2, mountaincar=3, acrobot=3)[config["env"]]


def main():
    envs_pre = []
    for tl in [1, 4, 16, 64, 128]:
        env = f"env=dsmdp;n_states=64;n_acts={4};d_obs=64;rdist=N;rpo=64;tl={tl}"
        envs_pre.append(env)
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

    config = {
        "n_seeds": 8,
        "agent": "linear_transformer",
        "env": None,
        "save_dir": None,
        "train": True,
        "n_envs": 64,
        "n_steps": 128,
        "total_timesteps": 100e6
    }
    configs = []
    for env_pre in envs_pre:
        c = config.copy()
        c["env"] = env_pre
        c["save_dir"] = f"../data/{env_pre}"
        configs.append(c)
    txt_pre = experiment_utils.create_command_txt_from_configs(configs, python_command='python train.py')

    config = {
        "n_seeds": 8,
        "agent": "linear_transformer",
        "env": None,
        "save_dir": None,
        "load_dir": None,
        "train": False,
        "n_envs": 64,
        "n_steps": 128,
        "total_timesteps": 100e6
    }

    configs = []
    for env_pre in envs_pre:
        for env_trans in envs_transfer:
            if get_n_acts(env_pre) != get_n_acts(env_trans):
                continue
            c = config.copy()
            c["env"] = env_trans
            c["save_dir"] = f"../data/transfer/{env_trans}/{env_pre}"
            c["load_dir"] = f"../data/{env_pre}"
            configs.append(c)
    txt_trans = experiment_utils.create_command_txt_from_configs(configs, python_command='python train.py')

    txt = txt_pre + txt_trans
    lines = [line for line in txt.split("\n") if line]
    out = []
    for i, line in enumerate(lines):
        out.append(f'CUDA_VISIBLE_DEVICES={i % 5} {line} &')
        if i % 5 == 4:
            out.append("wait")
    out = "\n".join(out)

    with open("train.sh", "w") as f:
        f.write(out)


if __name__ == '__main__':
    main()
