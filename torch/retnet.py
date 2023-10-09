import torch


def ret_parallel_forward(q, k, v):
    # q, k, v: B, H, T, D
    B, H, T, D = q.shape
    mask = torch.tril(torch.ones(T, T, dtype=bool))  # causal masking
    attn = mask * (q @ k.mT)
    out = attn @ v
    return out


def ret_recurrent_forward(q, k, v, state=None):
    # q, k, v: B, H, 1, D
    # state: H, D, D
    B, H, _, D = q.shape
    if state is None:
        state = torch.zeros(D, D)
    state = state + k.mT * v  # B, H, D, D
    out = q @ state
    return out, state


def main():
    torch.manual_seed(0)
    b, h, t, d = 16, 12, 100, 32
    q, k, v = torch.randn(b, h, t, d), torch.randn(b, h, t, d), torch.randn(b, h, t, d)

    # ------ PARALLEL FORWARD ------
    out1 = ret_parallel_forward(q, k, v)

    # ------ RECURRENT FORWARD ------
    out2 = torch.zeros_like(out1)
    state = None
    for t in range(q.shape[2]):
        out2[:, :, [t]], state = ret_recurrent_forward(q[:, :, [t]], k[:, :, [t]], v[:, :, [t]], state=state)

    print(f'{q.shape=} {k.shape=} {v.shape=}')
    print(f'{out1.shape=} {out2.shape=}')
    print(torch.allclose(out1, out2, atol=1e-4))


if __name__ == '__main__':
    main()

