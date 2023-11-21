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

def main():
    envs_pre = {}

    for n_acts in [2, 3, 4]:
        for tl in [4, 128]:
            denvu = f"env=dsmdp;n_states=64;n_acts={n_acts};d_obs=64;rdist=U;rpo=64;tl={tl}"
            denvn = f"env=dsmdp;n_states=64;n_acts={n_acts};d_obs=64;rdist=N;rpo=64;tl={tl}"

            cenv = f"env=csmdp;d_state=8;n_acts={n_acts};d_obs=64;delta=F;rpo=64;tl={tl}"
            cenvd = f"env=csmdp;d_state=8;n_acts={n_acts};d_obs=64;delta=T;rpo=64;tl={tl}"
            envs_pre[denvu] = n_acts
            envs_pre[denvn] = n_acts
            envs_pre[cenv] = n_acts
            envs_pre[cenvd] = n_acts

    envs_transfer = {
        "env=gridenv;grid_len=8;fobs=T;rpo=64;tl=128": 4,
        "env=cartpole;fobs=T;rpo=64;tl=128": 2,
        "env=mountaincar;fobs=T;rpo=64;tl=128": 3,
        "env=acrobot;fobs=T;rpo=64;tl=128": 3,
    }

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
            if envs_pre[env_pre] != envs_transfer[env_trans]:
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
        out.append(f'CUDA_VISIBLE_DEVICES={i%6} {line} &')
        if i % 6 == 5:
            out.append("wait")
    out = "\n".join(out)

    with open("train.sh", "w") as f:
        f.write(out)


if __name__ == '__main__':
    main()

