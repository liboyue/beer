import torch
import numpy as np

def identity(x, *args, **kwargs):
    # log.info('identity')
    return x.clone().detach()

# top_a
def top(x, a):
    dim = x.shape[0]
    if a == 0:
        return 0
    if a >= dim:
        return x
    index_array = xp.argpartition(x, kth=a, axis=0)[a:]
    xp.put_along_axis(x, index_array, 0, axis=0)
    return x

# Random_a compressor, keep a values
def random(x, a):
    dim = x.shape[0]
    if a == 0:
        return 0
    if a == dim:
        return x
    zero_mask = torch.randperm(dim, device=x.device)[:dim - a]
    x[zero_mask] = 0
    return x

def unbiased_random(x, a):
    dim = x.shape[0]
    return dim / a * random(x, a)


from beer import log
def gsgd(x, b):
    norm = torch.norm(x)
    if norm < 1e-10:
        log.info(norm)
        return x

    delta = np.sqrt(x.shape[0]) / (2 **(b - 1))
    tau = 1 + delta if delta > 1 else 1 + delta ** 2
    tmp = (2 ** (b - 1)) / norm * torch.abs(x) + torch.randn(x.shape, device=x.device)
    tmp = torch.max(tmp, torch.zeros(1, device=x.device))
    return torch.sign(x) * torch.floor(tmp) * (norm / (2 ** (b - 1)) / tau)


# random quantization 2-norm with level s
def random_quantization(x, s):
    # xnorm = xp.linalg.norm(x, axis=0)
    xnorm = torch.norm(x)
    if s == 0 or xnorm == 0:
        return xp.zeros(x.shape, dtype=int)
    noise = torch.randn(x.shape, device=x.device)
    rounded = torch.floor(s * torch.abs(x) / xnorm + noise)
    compressed = (xnorm / s) * torch.sign(x) * rounded
    return compressed


# natural compression (power of 2 for each coordinate)
def natural_compression(x):
    dim = x.shape[0]
    logx = xp.ma.log2(xp.abs(x)).filled(-15)
    logx_floor = xp.floor(logx)
    noise = xp.random.uniform(0.0, 1.0, dim)
    leftx = xp.exp2(logx_floor)
    rounded = xp.floor(xp.ma.log2(xp.abs(x) + leftx * noise).filled(-15))
    compressed = xp.sign(x) * xp.exp2(rounded)
    return compressed
