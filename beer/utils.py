import torch
import torch.distributed as dist
from torch.nn.modules import Module

from beer import log

def all_reduce_tensors(tensors, async_op=True):
    r"""Perform all-reduce on a list of tensors.

    Args:
        tensors (list):
            The list of tensors to reduce.

    Returns:
        list: Returns a list of request handlers.
    """
    return [dist.all_reduce(tensor, async_op=async_op) for tensor in tensors]


def reduce_tensors(tensors, dst, group, bufs=None, async_op=True):
    r"""Perform reduce on a list of tensors.

    Args:
        tensors:
            The list of tensors to reduce.

        dst:
            The destination rank.

        group:
            The desired communication group.

        bufs (optional):
            The buffers to store reduced parameters. If not provided,
            in-place operations will be performed on tensors.

    Returns:
        list: Returns a list of request handlers.
    """
    reqs = []

    if bufs is None:
        if tensors[0].device.type == 'cpu':
            for tensor in tensors:
                # Hack for Gloo on CPU. It may change the sender's tensor.
                if dist.get_backend() == 'gloo':
                    tensor = tensor.clone().detach()
                reqs.append(dist.reduce(tensor, dst, group=group, async_op=async_op))
        else:
            for i, tensor in enumerate(tensors):
                if dist.get_backend() == 'gloo':
                    tensor = tensor.clone().detach()
                reqs.append(dist.reduce(tensor, dst, group=group, async_op=async_op))
        # fi
    else:
        if tensors[0].device.type == 'cpu':
            for tensor, buf in zip(tensors, bufs):
                buf[:] = tensor[:]
                reqs.append(dist.reduce(buf, dst, group=group, async_op=async_op))
        else:
            for i, tensor in enumerate(tensors):
                buf = bufs[i]
                buf[:] = tensor[:]
                reqs.append(dist.reduce(buf, dst, group=group, async_op=async_op))
        # fi
    # fi
    return reqs

def flatten_tensors(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors], dim=0).detach()

def unflatten_tensors(flat_tensor, param_info):
    outputs = []

    offset = 0
    for index in range(len(param_info)):
        tensor = flat_tensor \
                .narrow(0, offset, param_info[index]['numel']) \
                .reshape(param_info[index]['shape'])
        outputs.append(tensor)
        offset += param_info[index]['numel']

    return outputs

def assign_unflattened_tensors(tensors, flat_tensors, param_info):
    new_tensors = unflatten_tensors(flat_tensors, param_info)
    for old, new in zip(tensors, new_tensors):
        tmp = old.data
        old.data = new.data
        del tmp
