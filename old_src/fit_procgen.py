import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default=None)
parser.add_argument("--n_iter", type=int, default=300000)
parser.add_argument("--bs", type=int, default=1024)
parser.add_argument("--lr", type=float, default=3e-4)

args = parser.parse_args()


def main(args):
    assert args.save_dir is not None

    with open(f"{args.save_dir}/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print({k: v.shape for k, v in dataset.items()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_obs, ds_logits = dataset['obs'], dataset['logits']
    ds_obs = torch.from_numpy(ds_obs).to(torch.float32).to(device)
    ds_logits = torch.from_numpy(ds_logits).to(torch.float32).to(device)

    print(ds_obs.shape, ds_logits.shape)
    print(ds_obs.dtype, ds_logits.dtype)

    agent = nn.Sequential(
        nn.Linear(ds_obs.shape[-1], 512),
        nn.GELU(),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Linear(256, 15),
    )

    agent = agent.to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)

    N, T, *_ = dataset['obs'].shape

    data = dict(ce=[], entr_tar=[])
    pbar = tqdm(range(args.n_iter))
    for i in pbar:
        i_n, i_t = torch.randint(0, N, (args.bs,)).to(device), torch.randint(0, T, (args.bs,)).to(device)

        x_batch = ds_obs[i_n, i_t]
        y_batch = ds_logits[i_n, i_t]
        logits = agent.forward(x_batch)
        probs = torch.nn.functional.softmax(y_batch, dim=-1)
        ce = nn.functional.cross_entropy(logits, probs, reduction='none')
        print(ce.shape, ce)
        loss = ce.sum(axis=-1).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        entr_tar = torch.distributions.Categorical(probs=probs).entropy().mean()
        data['ce'].append(loss.item())
        data['entr_tar'].append(entr_tar.item())
        pbar.set_postfix(loss=loss.item(), entr_tar=entr_tar.item())

    data = {k: np.array(v) for k, v in data.items()}
    with open(f"{args.save_dir}/train_stats.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main(args)
