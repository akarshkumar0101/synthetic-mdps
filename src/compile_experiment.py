import math
import os

import numpy as np

import experiment_utils
import icl_bc_ed
import icl_gen
import unroll

txt_header_main = "\n".join(["#!/bin/bash",
                             "source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps/bin/activate",
                             "cd /data/vision/phillipi/akumar01/synthetic-mdps/src", "\n" * 3])
txt_header_procgen = "\n".join(["#!/bin/bash",
                                "source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-procgen/bin/activate",
                                "cd /data/vision/phillipi/akumar01/synthetic-mdps/src", "\n" * 3])
txt_header_atari = "\n".join(["#!/bin/bash",
                              "source /data/vision/phillipi/akumar01/.virtualenvs/synthetic-mdps-atari/bin/activate",
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

np.random.seed(1)
envs_synthetic = []
for i in range(4):
    i_d, i_s, t_a, t_c, t_l, t_s, o_d, o_c, r_c = [np.random.randint(0, 5) for _ in range(9)]
    env_id = f"name=csmdp;i_d={i_d};i_s={i_s};t_a={t_a};t_c={t_c};t_l={t_l};t_s={t_s};o_d={o_d};o_c={o_c};r_c={r_c};tl=64"
    envs_synthetic.append(env_id)
    env_id = f"name=csmdp;i_d={i_d};i_s={i_s};t_a={t_a};t_c={t_c};t_l={t_l};t_s={t_s};o_d={o_d};o_c={o_c};r_c={r_c};tl=1"
    envs_synthetic.append(env_id)

for i in range(4):
    i_d, i_s, t_a, t_s, o_d = [np.random.randint(0, 5) for _ in range(5)]
    env_id = f"name=dsmdp;i_d={i_d};i_s={i_s};t_a={t_a};t_s={t_s};o_d={o_d};tl=64"
    envs_synthetic.append(env_id)
    env_id = f"name=dsmdp;i_d={i_d};i_s={i_s};t_a={t_a};t_s={t_s};o_d={o_d};tl=1"
    envs_synthetic.append(env_id)

# np.random.seed(1)
# for i in range(3):
#     t_a, t_c, o_d = [np.random.randint(0, 5) for _ in range(3)]
#     env_id = f"name=rf;t_a={t_a};t_c={t_c};o_d={o_d}"
#     envs_synthetic.append(env_id)

# envs_synthetic += ['zero_act']

envs_classic = [
    "name=CartPole-v1;tl=500",
    "name=Acrobot-v1;tl=500",
    "name=MountainCar-v0;tl=500",
    "name=DiscretePendulum-v1;tl=500",
]

envs_minatar = [
    "name=Asterix-MinAtar;tl=500",
    "name=Breakout-MinAtar;tl=500",
    "name=Freeway-MinAtar;tl=500",
    "name=SpaceInvaders-MinAtar;tl=500",
]

envs_train = envs_synthetic

envs_procgen = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist",
                "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

envs_atari_57 = ["Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis", "BankHeist", "BattleZone",
                 "BeamRider", "Berzerk", "Bowling", "Boxing", "Breakout", "Centipede", "ChopperCommand", "CrazyClimber",
                 "Defender", "DemonAttack", "DoubleDunk", "Enduro", "FishingDerby", "Freeway", "Frostbite", "Gopher",
                 "Gravitar", "Hero", "IceHockey", "Jamesbond", "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge",
                 "MsPacman", "NameThisGame", "Phoenix", "Pitfall", "Pong", "PrivateEye", "Qbert", "Riverraid",
                 "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris", "SpaceInvaders", "StarGunner", "Surround",
                 "Tennis", "TimePilot", "Tutankham", "UpNDown", "Venture", "VideoPinball", "WizardOfWor", "YarsRevenge",
                 "Zaxxon", ]
envs_atari_16 = ["Pong", "Breakout", "SpaceInvaders", "Asterix", "Amidar", "Freeway", "Boxing", "Jamesbond",
                 "Riverraid", "Hero", "Krull", "Tutankham", "Kangaroo", "MsPacman", "Defender", "BeamRider"]

envs_mujoco = ["Reacher", "Pusher", "InvertedPendulum", "InvertedDoublePendulum", "HalfCheetah", "Hopper", "Swimmer",
               "Walker2d", "Ant", "Humanoid", "HumanoidStandup"]

envs_dm_control = ["acrobot-swingup", "acrobot-swingup_sparse", "ball_in_cup-catch", "cartpole-balance",
                   "cartpole-balance_sparse", "cartpole-swingup", "cartpole-swingup_sparse", "cartpole-two_poles",
                   "cartpole-three_poles", "cheetah-run", "dog-stand", "dog-walk", "dog-trot", "dog-run", "dog-fetch",
                   "finger-spin", "finger-turn_easy", "finger-turn_hard", "fish-upright", "fish-swim", "hopper-stand",
                   "hopper-hop", "humanoid-stand", "humanoid-walk", "humanoid-run", "humanoid-run_pure_state",
                   "humanoid_CMU-stand", "humanoid_CMU-walk", "humanoid_CMU-run", "manipulator-bring_ball",
                   "manipulator-bring_peg", "manipulator-insert_ball", "manipulator-insert_peg", "pendulum-swingup",
                   "point_mass-easy", "point_mass-hard", "quadruped-walk", "quadruped-run", "quadruped-escape",
                   "quadruped-fetch", "reacher-easy", "reacher-hard", "stacker-stack_2", "stacker-stack_4",
                   "swimmer-swimmer6", "swimmer-swimmer15", "walker-stand", "walker-walk", "walker-run", ]

# envs_test = envs_classic
envs_test = envs_classic + envs_minatar
# envs_test = envs_classic + envs_minatar + envs_atari_16 + envs_procgen

dataset_dirs = {}
for env_id in envs_synthetic:
    dataset_dirs[env_id] = f"/synthetic/{env_id}/"
for env_id in envs_classic:
    dataset_dirs[env_id] = f"/real/classic/{env_id}/"
for env_id in envs_minatar:
    dataset_dirs[env_id] = f"/real/minatar/{env_id}/"
for env_id in envs_procgen:
    dataset_dirs[env_id] = f"/real/procgen/{env_id}/"
for env_id in envs_atari_57:
    dataset_dirs[env_id] = f"/real/atari/{env_id}/"

env_test2ft_steps = {
    "name=CartPole-v1": 100,
    "name=Acrobot-v1": 100,
    "name=MountainCar-v0": 100,
    "name=DiscretePendulum-v1": 100,
    "name=Asterix-MinAtar": 2000,
    "name=Breakout-MinAtar": 500,
    "name=Freeway-MinAtar": 2000,
    "name=SpaceInvaders-MinAtar": 500,
}


# def change_to_n_gpus(txt, n_gpus):
#     lines = [line for line in txt.split("\n") if line]
#     out = []
#     for i, line in enumerate(lines):
#         out.append(f'CUDA_VISIBLE_DEVICES={i % n_gpus} {line} &')
#         if i % n_gpus == n_gpus - 1 or i == len(lines) - 1:
#             out.append("wait")
#     out = "\n".join(out) + "\n"
#     return out


def exp_train(dir_exp, obj="bc", n_augs=0, n_iters_eval=100, n_iters=80000):
    cfg_default = vars(icl_bc_ed.parse_args())
    # print(cfg_default)

    cfgs = []

    # ---------------- PRETRAINING ON TRAIN ENVS ----------------
    for env_id in envs_train:
        cfg = cfg_default.copy()
        cfg.update(
            dataset_paths=f"{dir_exp}/datasets/{dataset_dirs[env_id]}/dataset.pkl",
            exclude_dataset_paths=None,
            save_dir=f"{dir_exp}/train_{obj}/{env_id}",
            n_iters=n_iters, n_iters_eval=n_iters_eval, obj=obj, n_ckpts=5, n_augs=n_augs,
            bs=64, ctx_len=1024,
        )
        cfgs.append(cfg)
    # ---------------- PRETRAINING ON TEST ENV ----------------
    for env_id in envs_test:
        cfg = cfg_default.copy()
        cfg.update(
            dataset_paths=f"{dir_exp}/datasets/{dataset_dirs[env_id]}/dataset.pkl",
            exclude_dataset_paths=None,
            save_dir=f"{dir_exp}/train_{obj}/{env_id}",
            n_iters=n_iters, n_iters_eval=n_iters_eval, obj=obj, n_ckpts=5, n_augs=n_augs,
            bs=64, ctx_len=1024,
        )
        cfgs.append(cfg)
    # ---------------- PRETRAINING ON ALL TEST ENVS ----------------
    cfg = cfg_default.copy()
    cfg.update(
        dataset_paths=f"{dir_exp}/datasets/real/*/*/dataset.pkl",
        exclude_dataset_paths=None,
        save_dir=f"{dir_exp}/train_{obj}/all",
        n_iters=n_iters, n_iters_eval=n_iters_eval, obj=obj, n_ckpts=5, n_augs=n_augs,
        bs=64, ctx_len=1024,
    )
    cfgs.append(cfg)
    # ---------------- PRETRAINING ON N-1 ENVS ----------------
    for env_id in envs_test:
        cfg = cfg_default.copy()
        cfg.update(
            dataset_paths=f"{dir_exp}/datasets/real/*/*/dataset.pkl",
            exclude_dataset_paths=f"{dir_exp}/datasets/{dataset_dirs[env_id]}/dataset.pkl",
            save_dir=f"{dir_exp}/train_{obj}/all-{env_id}",
            n_iters=n_iters, n_iters_eval=n_iters_eval, obj=obj, n_ckpts=5, n_augs=n_augs,
            bs=64, ctx_len=1024,
        )
        cfgs.append(cfg)

    txt = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_bc_ed.py')
    return txt


def exp_test(dir_exp, obj="bc"):
    cfg_default = vars(icl_bc_ed.parse_args())
    # print(cfg_default)
    # lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    # lrs = [3e-4]
    # percent_datas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    # percent_datas = [0.00025]
    lrs_pds = []
    # for lr in [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]:
    for lr in [3e-4]:
        lrs_pds.append((lr, 1.0))
    # for pd in [0.00025, 0.001, 0.01, 0.1]:
    #     lrs_pds.append((3e-4, pd))

    n_iters_eval = 200
    n_augs = 0

    cfgs = []
    for lr, pd in lrs_pds:
        for env_id_test in envs_test:
            # n_iters = env_test2ft_steps[env_id_test]
            n_iters = 2000
            # ---------------- TRAINED ON TRAIN/TEST ----------------
            for env_id_train in [*envs_train, env_id_test]:
                cfg = cfg_default.copy()
                cfg.update(
                    dataset_path=f'{dir_exp}/datasets/{dataset_dirs[env_id_test]}/dataset.pkl',
                    load_dir=f"{dir_exp}/train_{obj}/{env_id_train}",
                    save_dir=f"{dir_exp}/test_{obj}/{env_id_test}/{env_id_train}_lr={lr}_pd={pd}",
                    n_iters=n_iters, n_iters_eval=n_iters_eval, obj=obj, n_augs=n_augs,
                    # save_agent=True, n_ckpts=5,
                    lr=lr, percent_data=pd,
                )
                cfgs.append(cfg)
            # ---------------- TRAINED ON ALL ----------------
            cfg = cfg_default.copy()
            cfg.update(
                dataset_path=f'{dir_exp}/datasets/{dataset_dirs[env_id_test]}/dataset.pkl',
                load_dir=f"{dir_exp}/train_{obj}/all",
                save_dir=f"{dir_exp}/test_{obj}/{env_id_test}/all_lr={lr}_pd={pd}",
                n_iters=n_iters, n_iters_eval=n_iters_eval, obj=obj, n_augs=n_augs,
                # save_agent=True, n_ckpts=5,
                lr=lr, percent_data=pd,
            )
            cfgs.append(cfg)

            # ---------------- TRAINED ON N-1 ----------------
            cfg = cfg_default.copy()
            cfg.update(
                dataset_path=f'{dir_exp}/datasets/{dataset_dirs[env_id_test]}/dataset.pkl',
                load_dir=f"{dir_exp}/train_{obj}/all-{env_id_test}",
                save_dir=f"{dir_exp}/test_{obj}/{env_id_test}/n-1_lr={lr}_pd={pd}",
                n_iters=n_iters, n_iters_eval=n_iters_eval, obj=obj, n_augs=n_augs,
                # save_agent=True, n_ckpts=5,
                lr=lr, percent_data=pd,
            )
            cfgs.append(cfg)

            # ---------------- FROM-SCRATCH ----------------
            cfg = cfg_default.copy()
            cfg.update(
                dataset_path=f'{dir_exp}/datasets/{dataset_dirs[env_id_test]}/dataset.pkl',
                load_dir=None,
                save_dir=f"{dir_exp}/test_{obj}/{env_id_test}/scratch_lr={lr}_pd={pd}",
                n_iters=n_iters, n_iters_eval=n_iters_eval, obj=obj, n_augs=n_augs,
                # save_agent=True, n_ckpts=5,
                lr=lr, percent_data=pd,
            )
            cfgs.append(cfg)

    txt = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_bc_ed.py')
    return txt


def exp_unroll(dir_exp, obj="bc"):
    cfg_default = vars(unroll.parser.parse_args())
    # print(cfg_default)

    cfgs = []
    for env_id_test in envs_test:
        for ckpt in [0, 1, 2, 3, 4, "final"]:
            # ---------------- TRAINED ON TRAIN/TEST ----------------
            for env_id_train in [*envs_train, env_id_test]:
                cfg = cfg_default.copy()
                cfg.update(
                    dataset_path=f'{dir_exp}/datasets/{dataset_dirs[env_id_test]}/dataset.pkl',
                    env_id=env_id_test,
                    ckpt_path=f"{dir_exp}/test_{obj}/{env_id_test}/{env_id_train}/ckpt_{ckpt}.pkl",
                )
                cfgs.append(cfg)
            # ---------------- TRAINED ON ALL ----------------
            cfg = cfg_default.copy()
            cfg.update(
                dataset_path=f'{dir_exp}/datasets/{dataset_dirs[env_id_test]}/dataset.pkl',
                env_id=env_id_test,
                ckpt_path=f"{dir_exp}/test_{obj}/{env_id_test}/all/ckpt_{ckpt}.pkl",
            )
            cfgs.append(cfg)

            # ---------------- TRAINED ON N-1 ----------------
            cfg = cfg_default.copy()
            cfg.update(
                dataset_path=f'{dir_exp}/datasets/{dataset_dirs[env_id_test]}/dataset.pkl',
                env_id=env_id_test,
                ckpt_path=f"{dir_exp}/test_{obj}/{env_id_test}/n-1/ckpt_{ckpt}.pkl",
            )
            cfgs.append(cfg)

            # ---------------- FROM-SCRATCH ----------------
            cfg = cfg_default.copy()
            cfg.update(
                dataset_path=f'{dir_exp}/datasets/{dataset_dirs[env_id_test]}/dataset.pkl',
                env_id=env_id_test,
                ckpt_path=f"{dir_exp}/test_{obj}/{env_id_test}/scratch/ckpt_{ckpt}.pkl",
            )
            cfgs.append(cfg)

    txt = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python unroll.py')
    return txt


def exp_data_syn(dir_exp):
    cfg_default = vars(icl_gen.parse_args())
    # ------------------- SYNTHETIC -------------------
    cfgs = []
    for env_id in envs_synthetic:
        cfg = cfg_default.copy()
        cfg.update(
            env_id=env_id,
            agent_id="small",
            n_seeds_seq=32,
            n_seeds_par=32,
            n_iters_train=100,
            n_iters_eval=16,
            lr=3e-4,
            save_dir=f"{dir_exp}/datasets/{dataset_dirs[env_id]}/",
        )
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_gen_ed.py')
    return txt


def exp_data_classic(dir_exp):
    cfg_default = vars(icl_gen.parse_args())
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
            n_envs=128,
            n_iters_eval=64,
            lr=3e-4,
            best_of_n_experts=50,
            save_dir=f"{dir_exp}/datasets/{dataset_dirs[env_id]}/",
        )
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_gen_ed.py')
    return txt


def exp_data_minatar(dir_exp):
    cfg_default = vars(icl_gen.parse_args())
    # ------------------- MINATAR -------------------
    cfgs = []
    for env_id in envs_minatar:
        cfg = cfg_default.copy()
        cfg.update(
            env_id=env_id,
            agent_id="minatar",
            n_seeds_seq=1,
            n_seeds_par=1,
            n_envs=128,
            n_envs_batch=8,
            n_updates=32,
            gamma=.999,
            n_iters_train=2000,
            n_iters_eval=64,
            lr=1e-3,
            best_of_n_experts=10,
            save_dir=f"{dir_exp}/datasets/{dataset_dirs[env_id]}/",
        )
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default, python_command='python icl_gen_ed.py')
    return txt


def exp_data_atari(dir_exp):
    cfg_default = dict(track=False, env_id="none", save_dir="none")
    # ------------------- ATARI -------------------
    cfgs = []
    for env_id in envs_atari_16:
        cfg = cfg_default.copy()
        cfg.update(
            track=True,
            env_id=f"{env_id}-v5",
            save_dir=f"{dir_exp}/datasets/{dataset_dirs[env_id]}/",
        )
        cfgs.append(cfg)
    txt_atari = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default,
                                                                 python_command='python ppo_atari_envpool.py')
    return txt_atari


def exp_data_procgen(dir_exp):
    cfg_default = dict(track=False, env_id="none", save_dir="none")
    # ------------------- PROCGEN -------------------
    cfgs = []
    for env_id in envs_procgen:
        cfg = cfg_default.copy()
        cfg.update(
            track=True,
            env_id=env_id,
            save_dir=f"{dir_exp}/datasets/{dataset_dirs[env_id]}/",
        )
        cfgs.append(cfg)
    txt_procgen = experiment_utils.create_command_txt_from_configs(cfgs, cfg_default,
                                                                   python_command='python ppo_procgen.py')

    return txt_procgen


def write_to_nodes_gpus(file, txt, n_nodes=1, n_gpus=1, txt_header=txt_header_main):
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

    basename = os.path.basename(file).split(".")[0]
    dirname = os.path.dirname(file)
    with open(f"{file}", "w") as f:
        f.write(txt)
    os.makedirs(f"{dirname}/{basename}", exist_ok=True)
    for i_node, txt_node in enumerate(txt_nodes):
        with open(f"{dirname}/{basename}/{i_node}.sh", "w") as f:
            f.write(txt_node)


def create_agent_atari(dir_exp):
    cfgs = []
    for env_id in envs_atari_57:
        cfg = dict(env_id=f"{env_id}-v5", save_dir=f"{dir_exp}/datasets/atari/{env_id}/")
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs, python_command='python cleanrl/ppo_atari_envpool.py')
    return txt


def create_agent_procgen(dir_exp):
    cfgs = []
    for env_id in envs_procgen:
        cfg = dict(env_id=f"{env_id}", save_dir=f"{dir_exp}/datasets/procgen/{env_id}/")
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs, python_command='python cleanrl/ppg_procgen.py')
    return txt


def create_agent_mujoco(dir_exp):
    cfgs = []
    for env_id in envs_mujoco:
        rpo_alpha = 0.5
        if env_id in ["Ant", "HalfCheetah", "Hopper", "InvertedDoublePendulum", "Reacher", "Swimmer", "Pusher"]:
            rpo_alpha = 0.01
        cfg = dict(env_id=f"{env_id}-v4", save_dir=f"{dir_exp}/datasets/mujoco/{env_id}/", rpo_alpha=rpo_alpha)
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs,
                                                           python_command='python cleanrl/rpo_continuous_action.py')
    return txt


def create_agent_dm_control(dir_exp):
    cfgs = []
    for env_id in envs_dm_control:
        cfg = dict(env_id=f"dm_control/{env_id}-v0", save_dir=f"{dir_exp}/datasets/dm_control/{env_id}/")
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs,
                                                           python_command='python cleanrl/rpo_continuous_action.py')
    return txt


def collect_data_atari(dir_exp):
    return ""


def collect_data_procgen_sweep_embed_name(dir_exp):
    cfgs = []

    for env_id in envs_procgen[:3]:
        for embed_name in ["resnet34_layer2_avg", "resnet34_layer3_avg", "resnet34_layer4_avg", "resnet34_layer2_max",
                           "resnet34_layer3_max", "resnet34_layer4_max"]:
            cfg = dict(env_id=f"{env_id}",
                       load_dir=f"{dir_exp}/datasets/procgen/{env_id}/",
                       save_dir=f"{dir_exp}/datasets_temp/procgen/{env_id}/{embed_name}",
                       num_steps=131072 // 8,
                       embed_name=embed_name)
            cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs,
                                                           python_command='python cleanrl/collect_ppg_procgen.py')
    return txt


def collect_data_procgen_sweep_embed_name_train(dir_exp):
    cfgs = []

    for env_id in envs_procgen[:3]:
        for embed_name in ["resnet34_layer2_avg", "resnet34_layer3_avg", "resnet34_layer4_avg", "resnet34_layer2_max",
                           "resnet34_layer3_max", "resnet34_layer4_max"]:
            cfg = dict(save_dir=f"{dir_exp}/datasets_temp/procgen/{env_id}/{embed_name}")
            cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs, python_command='python fit_procgen.py')
    return txt


def collect_data_procgen(dir_exp):
    cfgs = []
    for env_id in envs_procgen:
        cfg = dict(env_id=f"{env_id}",
                   load_dir=f"{dir_exp}/datasets/procgen/{env_id}/",
                   save_dir=f"{dir_exp}/datasets/procgen/{env_id}/",
                   num_steps=131072,
                   embed_name="resnet34_layer4_avg")
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs,
                                                           python_command='python cleanrl/collect_ppg_procgen.py')
    return txt


def collect_data_mujoco(dir_exp):
    cfgs = []
    for env_id in envs_mujoco:
        cfg = dict(env_id=f"{env_id}-v4",
                   load_dir=f"{dir_exp}/datasets/mujoco/{env_id}/",
                   save_dir=f"{dir_exp}/datasets/mujoco/{env_id}/",
                   num_steps=131072)
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs,
                                                           python_command='python cleanrl/collect_rpo_continuous_action.py')
    return txt


def collect_data_dm_control(dir_exp):
    cfgs = []
    for env_id in envs_dm_control:
        # TODO: quadruped-escape needs SyncVectorEnv
        cfg = dict(env_id=f"dm_control/{env_id}-v0",
                   load_dir=f"{dir_exp}/datasets/dm_control/{env_id}/",
                   save_dir=f"{dir_exp}/datasets/dm_control/{env_id}/",
                   num_steps=13107)
        cfgs.append(cfg)
    txt = experiment_utils.create_command_txt_from_configs(cfgs,
                                                           python_command='python cleanrl/collect_rpo_continuous_action.py')
    return txt


if __name__ == '__main__':
    dir_exp = "/data/vision/phillipi/akumar01/synthetic-mdps-data"
    n_nodes, n_gpus = 2, 8
    os.system("rm -rf ./experiment/")
    os.makedirs("./experiment/", exist_ok=True)

    with open("./experiment/create_agent_atari.sh", "w") as f:
        f.write(create_agent_atari(dir_exp))
    with open("./experiment/create_agent_procgen.sh", "w") as f:
        f.write(create_agent_procgen(dir_exp))
    with open("./experiment/create_agent_mujoco.sh", "w") as f:
        f.write(create_agent_mujoco(dir_exp))
    with open("./experiment/create_agent_dm_control.sh", "w") as f:
        f.write(create_agent_dm_control(dir_exp))

    with open("./experiment/collect_data_atari.sh", "w") as f:
        f.write(collect_data_atari(dir_exp))
    with open("./experiment/collect_data_procgen.sh", "w") as f:
        f.write(collect_data_procgen(dir_exp))
    with open("./experiment/collect_data_mujoco.sh", "w") as f:
        f.write(collect_data_mujoco(dir_exp))
    with open("./experiment/collect_data_dm_control.sh", "w") as f:
        f.write(collect_data_dm_control(dir_exp))

    with open("./experiment/collect_data_procgen_sweep_embed_name.sh", "w") as f:
        f.write(collect_data_procgen_sweep_embed_name(dir_exp))
    with open("./experiment/collect_data_procgen_sweep_embed_name_train.sh", "w") as f:
        f.write(collect_data_procgen_sweep_embed_name_train(dir_exp))

    txt = exp_data_syn(dir_exp)
    write_to_nodes_gpus("./experiment/data_syn.sh", txt, n_nodes, n_gpus)
    txt = exp_data_classic(dir_exp)
    write_to_nodes_gpus("./experiment/data_classic.sh", txt, n_nodes, n_gpus)
    txt = exp_data_minatar(dir_exp)
    write_to_nodes_gpus("./experiment/data_minatar.sh", txt, n_nodes, n_gpus)
    # txt = exp_data_atari(dir_exp)
    # write_to_nodes_gpus("./experiment/data_atari.sh", txt, n_nodes, n_gpus, txt_header=txt_header_atari)
    # txt = exp_data_procgen(dir_exp)
    # write_to_nodes_gpus("./experiment/data_procgen.sh", txt, n_nodes, n_gpus, txt_header=txt_header_procgen)

    txt = exp_train(dir_exp, obj="bc", n_augs=int(1e6))
    write_to_nodes_gpus("./experiment/train_bc.sh", txt, n_nodes, n_gpus)

    txt = exp_test(dir_exp, obj="bc")
    write_to_nodes_gpus("./experiment/test_bc.sh", txt, n_nodes, n_gpus)

    # txt = exp_unroll(dir_exp, obj="bc")
    # write_to_nodes_gpus("./experiment/unroll_bc.sh", txt, n_nodes=8, n_gpus=4)

    # txt = exp_train(dir_exp, obj="wm")
    # write_to_nodes_gpus("./experiment/train_wm.sh", txt, n_nodes, n_gpus)
    #
    # txt = exp_test(dir_exp, obj="wm")
    # write_to_nodes_gpus("./experiment/test_wm.sh", txt, n_nodes, n_gpus)
