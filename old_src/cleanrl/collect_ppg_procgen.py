# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppg/#ppg_procgenpy
import os
import pickle
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.optim as optim
import torchvision
import tyro
from procgen import ProcgenEnv
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from tqdm.auto import tqdm

from ppg_procgen import Agent


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    # env_id: str = "starpilot"
    env_id: str = "bigfish"
    """the id of the environment"""
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 4096
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    adv_norm_fullbatch: bool = True
    """Toggle full batch advantage normalization as used in PPG code"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # PPG specific arguments
    n_iteration: int = 32
    """N_pi: the number of policy update in the policy phase """
    e_policy: int = 1
    """E_pi: the number of policy update in the policy phase """
    v_value: int = 1
    """E_V: the number of policy update in the policy phase """
    e_auxiliary: int = 6
    """E_aux:the K epochs to update the policy"""
    beta_clone: float = 1.0
    """the behavior cloning coefficient"""
    num_aux_rollouts: int = 4
    """the number of mini batch in the auxiliary phase"""
    n_aux_grad_accum: int = 1
    """the number of gradient accumulation in mini batch"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_phases: int = 0
    """the number of phases (computed in runtime)"""
    aux_batch_rollouts: int = 0
    """the number of rollouts in the auxiliary phase (computed in runtime)"""

    # embed_name: str = "resnet18_layer4_avg"
    # embed_name: str = "resnet34_layer3_max"
    embed_name: str = "resnet34_layer4_avg"
    load_dir: str = None
    save_dir: str = None


class ResNetEmbedder:
    def __init__(self, embed_name, device=None):  # ex. resnet18_layer4_avg
        self.resnet_type, self.layer, self.pool_type = embed_name.split("_")

        self.rn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.rn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.resize_transform = torchvision.transforms.Resize((224, 224))

        if self.resnet_type == "resnet18":
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        elif self.resnet_type == "resnet34":
            self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).to(device)
        elif self.resnet_type == "resnet50":
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        elif self.resnet_type == "resnet101":
            self.resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).to(device)
        else:
            raise NotImplementedError

        self.activation = {}
        self.resnet.layer1.register_forward_hook(self.get_activation('layer1'))
        self.resnet.layer2.register_forward_hook(self.get_activation('layer2'))
        self.resnet.layer3.register_forward_hook(self.get_activation('layer3'))
        self.resnet.layer4.register_forward_hook(self.get_activation('layer4'))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    @torch.no_grad()
    def embed_obs(self, x):  # bchw
        x = x.permute((0, 3, 1, 2))  # "bhwc" -> "bchw"
        x = self.resize_transform(x)  # rescale to 224x224 with torchvision
        x = x / 255.0
        x = (x - self.rn_mean[:, None, None]) / (self.rn_std[:, None, None] + 1e-8)
        _ = self.resnet(x)

        if self.pool_type == "avg":
            features = self.activation[self.layer].mean(dim=(-1, -2))
        elif self.pool_type == "max":
            features = self.activation[self.layer].max(dim=-1).values.max(dim=-1).values
        else:
            raise NotImplementedError
        return features


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.save_dir is not None and args.load_dir is not None
    args.save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    args.load_dir = os.path.abspath(os.path.expanduser(args.load_dir))

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_phases = int(args.num_iterations // args.n_iteration)
    args.aux_batch_rollouts = int(args.num_envs * args.n_iteration)
    assert args.v_value == 1, "Multiple value epoch (v_value != 1) is not supported yet"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_id, num_levels=0, start_level=0,
                      distribution_mode="easy")
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(f"{args.load_dir}/model.pth"))
    embedder = ResNetEmbedder(args.embed_name, device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)

    # ALGO Logic: Storage setup
    dummy_obs = torch.Tensor(envs.reset()).to(device)
    embed_obs_shape = embedder.embed_obs(dummy_obs).shape[1:]

    obs = np.zeros((args.num_steps, args.num_envs) + embed_obs_shape, dtype=np.float32)
    actions = np.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logits = np.zeros((args.num_steps, args.num_envs) + (envs.single_action_space.n,))
    logprobs = np.zeros((args.num_steps, args.num_envs))
    rewards = np.zeros((args.num_steps, args.num_envs))
    dones = np.zeros((args.num_steps, args.num_envs))
    values = np.zeros((args.num_steps, args.num_envs))
    # aux_obs = torch.zeros(
    #     (args.num_steps, args.aux_batch_rollouts) + envs.single_observation_space.shape, dtype=torch.uint8
    # )  # Saves lot system RAM
    # aux_returns = torch.zeros((args.num_steps, args.aux_batch_rollouts))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    train_stats = dict(global_step=[], episode_return=[], episode_length=[])

    pbar = tqdm(range(0, args.num_steps))
    for step in pbar:
        global_step += 1 * args.num_envs
        obs[step] = embedder.embed_obs(next_obs).cpu().numpy()
        dones[step] = next_done.cpu().numpy()

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value, logits_i = agent.get_action_and_value(next_obs)
            values[step] = value.flatten().cpu().numpy()
        actions[step] = action.cpu().numpy()
        logits[step] = logits_i.cpu().numpy()
        logprobs[step] = logprob.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1).cpu().numpy()
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        for item in info:
            if "episode" in item.keys():
                # print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                train_stats["global_step"].append(global_step)
                train_stats["episode_return"].append(item["episode"]["r"])
                train_stats["episode_length"].append(item["episode"]["l"])
                ret_mean, len_mean, = np.mean(train_stats["episode_return"]), np.mean(train_stats["episode_length"])
                pbar.set_postfix(ret=ret_mean, len=len_mean)

    print("Average episode return: ", np.mean(train_stats["episode_return"]))
    print("Average episode length: ", np.mean(train_stats["episode_length"]))

    print(f"Saving to {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    with open(f"{args.save_dir}/dataset_stats.pkl", "wb") as f:
        pickle.dump(train_stats, f)

    dataset = dict(obs=obs, act=actions, logits=logits, rew=rewards, done=dones)
    # dataset = {k: v.cpu().numpy() for k, v in dataset.items()}
    dataset = {k: np.swapaxes(v, 0, 1) for k, v in dataset.items()}
    dataset['obs'] = dataset['obs'].astype(np.float32)
    print("Dataset shape: ")
    print({k: v.shape for k, v in dataset.items()})
    with open(f"{args.save_dir}/dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    envs.close()
    writer.close()
