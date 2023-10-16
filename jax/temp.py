
import torch
from torch import nn

def method(n_models):
    models = [nn.Linear(20, 30) for _ in range(n_models)]
    return torch.func.stack_module_state(models)


a = method(10)
for k, v in a[0].items():
    print(k, v.shape)

# method = torch.vmap(method)
#
# a = method(torch.tensor([10, 10]))
# for k, v in a[0].items():
#     print(k, v.shape)



