import torch

import torch
from torch import nn


def create_mlp():
    return nn.Sequential(
        nn.Linear(10, 8),
        nn.Sigmoid(),
        nn.Linear(8, 8),
        nn.Sigmoid(),
        nn.Linear(8, 4),
        nn.Softmax(dim=-1),
    )


def main():
    nets = [create_mlp() for _ in range(5)]
    params, buffers = torch.func.stack_module_state(nets)

    def forward(params, buffers, x):
        return torch.func.functional_call(nets[0], (params, buffers), x)

    forward = torch.vmap(forward)
    x = torch.randn(5, 10)
    y = forward(params, buffers, x)

    print(y.shape)




if __name__ == '__main__':
    main()



