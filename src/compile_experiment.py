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

    envs = [
        "CartPole-v1",
        "Acrobot-v1",
        "synthetic"
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

    # ------------------- TRAINING -------------------
    cfgs = []
    cfg_shared = cfg_default.copy()
    cfg_shared.update(n_iters=10000)
    for time_perm in [True, False]:
        for env_id in envs:
            cfg = cfg_shared.copy()
            cfg.update(
                name=f"pretrain_{env_id}_time_perm_{time_perm}",
                dataset_path=f'../data/temp/expert_data_{env_id}.pkl',
                save_dir=f"{dir_exp}/pretrain/{env_id}_{time_perm}",
                time_perm=time_perm
            )
            cfgs.append(cfg)
    txt_pretrain = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default,
                                                                    python_command='python icl_bc.py')

    # ------------------- EVAL -------------------
    cfgs = []
    cfg_shared = cfg_default.copy()
    cfg_shared.update(n_iters=1000)
    for time_perm in [True, False]:
        for env_id_train in envs:
            for env_id_test in envs:
                cfg = cfg_shared.copy()
                cfg.update(
                    name=f"train_{env_id_train}_test_{env_id_test}_time_perm_{time_perm}",
                    dataset_path=f'../data/temp/expert_data_{env_id_test}.pkl',
                    load_dir=f"{dir_exp}/pretrain/{env_id_train}_{time_perm}",
                    save_dir=f"{dir_exp}/eval/{env_id_train}/{env_id_test}_{time_perm}",
                    time_perm=time_perm
                )
                cfgs.append(cfg)
    txt_eval = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default,
                                                                python_command='python icl_bc.py')

    # ------------------- FINE-TUNE EVAL: BC -------------------
    # cfgs = []
    # cfg_shared = bc_cfg_default.copy()
    # cfg_shared.update(n_seeds=32, agent_id=agent_id, run='train',
    #                   save_agent_params=False, n_envs=4, n_envs_batch=4, n_iters=300,  # 300
    #                   reset_layers='last', ft_layers='last')
    # for env_test in envs_test:
    #     for env_train in envs_train:
    #         cfg = cfg_shared.copy()
    #         cfg["env_id"] = env_test
    #         cfg["load_dir"] = f"{dir_exp}/pretrain/{env_train}"
    #         cfg["load_dir_teacher"] = f"{dir_exp}/expert/{env_test}"
    #         cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{env_train}"
    #         cfgs.append(cfg)
    #
    #     # random agent - no train
    #     cfg = cfg_shared.copy()
    #     cfg["env_id"] = env_test
    #     cfg["load_dir"] = None
    #     cfg["load_dir_teacher"] = f"{dir_exp}/expert/{env_test}"
    #     cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{'random_agent'}"
    #     cfg["lr"] = 0.0
    #     cfgs.append(cfg)
    #
    #     # random agent - train
    #     cfg = cfg_shared.copy()
    #     cfg["env_id"] = env_test
    #     cfg["load_dir"] = None
    #     cfg["load_dir_teacher"] = f"{dir_exp}/expert/{env_test}"
    #     cfg["save_dir"] = f"{dir_exp}/test/{env_test}/{'train_random'}"
    #     cfgs.append(cfg)
    # txt_eval = experiment_utils.create_command_txt_from_configs(cfgs, bc_cfg_default, python_command='python run_bc.py')

    txt_pretrain = change_to_n_gpus(txt_pretrain, n_gpus)
    txt_eval = change_to_n_gpus(txt_eval, n_gpus)
    txt = f"{txt_header}\n\n{txt_pretrain}\n{txt_eval}"
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
    #     f.write(experiment("../data/exp_iclbc/", 6))

    with open("experiment.sh", "w") as f:
        f.write(exp_generate_data("../data/exp_iclbc/", 6))
