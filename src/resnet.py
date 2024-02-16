import time

import torch
from torchvision.models import resnet18, ResNet18_Weights


def main():
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to('cuda')
    print(resnet)
    start_time = time.time()
    for i in range(1):
        x = torch.randn(64, 3, 64, 64).cuda()

        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        resnet.layer1.register_forward_hook(get_activation('layer1'))
        resnet.layer2.register_forward_hook(get_activation('layer2'))
        resnet.layer3.register_forward_hook(get_activation('layer3'))
        resnet.layer4.register_forward_hook(get_activation('layer4'))
        with torch.no_grad():
            y = resnet(x)
        print(x.shape, y.shape)
        print(activation['layer1'].shape)
        print(activation['layer2'].shape)
        print(activation['layer3'].shape)
        print(activation['layer4'].shape)
        end_time = time.time()
        print((64 * (i + 1)) / (end_time - start_time))


if __name__ == "__main__":
    main()
