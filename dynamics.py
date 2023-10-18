import torch


def main():
    A = torch.randn(10, 10)

    x = torch.randn(10)
    for i in range(1000):
        x = A @ x
        x = x / x.norm()
        print(x.tolist()[:10])


if __name__ == '__main__':
    main()
