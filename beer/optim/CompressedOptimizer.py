import torch
from torch.optim import Optimizer
from beer.utils import reduce_tensors, flatten_tensors
from beer import log

from . import compressor

class CompressedOptimizer(Optimizer):
    def __init__(self, model, lr=1, G=None, world_size=1, rank=0, compression_type=None, compression_params=[], **kwargs):
        super().__init__(model.module.parameters(), dict())
        log.info(f'world size = {world_size}, rank = {rank}, lr = {lr}, compression = {compression_type}({compression_params})')
        self.lr = lr
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.compression_type = compression_type
        self.compression_params = compression_params

        self.compression_operator = lambda x: getattr(compressor, compression_type)(x, *self.compression_params)
        self.G = G

    @torch.no_grad()
    def mix(self, flat_tensor, flat_buf):
        log.debug('Mixing')

        reqs = []

        for dst in range(self.world_size):
            log.debug('dst is rank %d', dst)
            group = self.G.process_group[dst]
            if dst == self.rank:
                # Recv
                log.debug('receiving')
                reqs += reduce_tensors([flat_tensor], dst, group, bufs=[flat_buf])
                log.debug('rank %d recv ', self.rank)
            else:
                # Send
                if self.rank in self.G.graph.neighbors(dst):
                    log.debug('sending to %d', dst)
                    reqs += reduce_tensors([flat_tensor], dst, group)
                    log.debug('rank %d send to %d', self.rank, dst)

        for req in reqs:
            req.wait()

        n_neighbors = len(list(self.G.neighbors(self.rank)))
        flat_buf.div_(n_neighbors + 1)

        log.debug('Mixing done')

    @torch.no_grad()
    def zero_grad(self):
        self.model.zero_grad()


    @torch.no_grad()
    def flatten_grads(self, module):
        return flatten_tensors([t.grad for t in module.parameters()])

