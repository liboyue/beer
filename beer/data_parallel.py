import torch
import torch.distributed as dist
from torch.nn.modules import Module

from copy import deepcopy
from beer.utils import flatten_tensors, assign_unflattened_tensors, all_reduce_tensors

from beer import log

class DataParallel(Module):
    r"""The base distributed data parallel module.

    To reduce memory copy, flatten tensors into buckets, then assign unflattened
    new tensor to parameters.

    .. note::
        The actual communication happens at the beginning of each forward call.
        When training, the model should be validated before optimizer.step() to
        produce correct results.
    """

    def __init__(self, module, use_ref_module=False):

        super().__init__()

        log.info('Using %s', self.__class__.__name__)

        self.param_info = [{'numel': param.numel(), 'shape': param.shape} for param in module.parameters()]

        self.device = next(module.parameters()).device
        self.module = module
        self.val_module = deepcopy(self.module).to(self.device)
        self.training = True

        self.flat_parameters = flatten_tensors(list(self.module.parameters())).to(self.device)
        assign_unflattened_tensors(self.module.parameters(), self.flat_parameters, self.param_info)

        self.flat_val_parameters = flatten_tensors(list(self.val_module.parameters())).to(self.device)
        assign_unflattened_tensors(self.val_module.parameters(), self.flat_val_parameters, self.param_info)

        log.info(f'Model dimension {self.flat_parameters.shape[0]}')
        with torch.no_grad():
            dist.broadcast(self.flat_parameters, 0)
        log.info('Broadcasting init params done')

        if use_ref_module:
            self.ref_module = deepcopy(self.module).to(self.device)
            self.flat_ref_parameters = flatten_tensors(list(self.ref_module.parameters())).to(self.device)
            assign_unflattened_tensors(self.ref_module.parameters(), self.flat_ref_parameters, self.param_info)
        else:
            self.ref_module = None

    @torch.no_grad()
    def eval(self):
        self.training = False
        self.flat_val_parameters[:] = self.flat_parameters[:]
        reqs = all_reduce_tensors([self.flat_val_parameters], async_op=False)
        self.flat_val_parameters /= dist.get_world_size()

        if self.ref_module is not None:
            self.ref_module.eval()

        return self.module.eval()

    @torch.no_grad()
    def train(self):
        self.training = True
        if self.ref_module is not None:
            self.ref_module.train()
        return self.module.train()

    def forward(self, *args, **kwargs):
        if self.training:
            if self.ref_module is not None:
                return self.module(*args, **kwargs), self.ref_module(*args, **kwargs)
            return self.module(*args, **kwargs)
        else:
            return self.val_module(*args, **kwargs)

    @torch.no_grad()
    def zero_grad(self):
        if self.training:
            self.module.zero_grad()
            if self.ref_module is not None:
                self.ref_module.zero_grad()
        else:
            self.val_module.zero_grad()

    @torch.no_grad()
    def zero_(self):
        if self.training:
            for p in self.module.parameters():
                p.zero_()
            if self.ref_module is not None:
                for p in self.ref_module.parameters():
                   p.zero_()
        else:
            for p in self.val_module.parameters():
                p.zero_()
