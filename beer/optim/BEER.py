import torch
from beer.optim import CompressedOptimizer
from beer.utils import flatten_tensors

from beer import log

class BEER(CompressedOptimizer):
    def __init__(self, model, gamma=0.1, **kwargs):
        super().__init__(model, **kwargs)
        self.gamma = gamma

        self.device = next(model.module.parameters()).device

        self.buf = torch.zeros_like(model.flat_parameters, device=self.device)
        self.H = torch.zeros_like(model.flat_parameters, device=self.device)
        self.V = torch.zeros_like(model.flat_parameters, device=self.device)
        self._G = torch.zeros_like(model.flat_parameters, device=self.device)

    def init(self):
        X = self.model.flat_parameters
        X -= self.lr * self.V
        self.H += self.compression_operator(X - self.H)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Original BEER
        # X^{t + 1} = X^t + gamma * H^t (W - I) - eta * V^t
        # Q_h^{t + 1} = c(X^{t + 1} - H^t)
        # H^{t + 1} = H^t + Q_h^{t + 1}
        # V^{t + 1} = V^t + gamma * G^t (W - I) + grad(X^{t + 1}) - grad(X^t)
        # Q_g^{t + 1} = c(V^{t + 1} - G^t)
        # G^{t + 1} = G^t + Q_g^{t + 1}

        # Reorder
        # X^{t + 1} = X^t + gamma * H^t (W - I) - eta * V^t
        # H^{t + 1} = H^t + c(X^{t + 1} - H^t)
        # V^{t + 1} = V^t + gamma * G^t (W - I) + grad(X^{t + 1}) - grad(X^t)
        # G^{t + 1} = G^t + c(V^{t + 1} - G^t)

        # Reorder
        # V^{t} = V^{t - 1} + gamma * G^{t - 1} (W - I) + grad(X^{t}) - grad(X^{t - 1})
        # G^{t} = G^{t - 1} + c(V^{t} - G^{t - 1})
        # X^{t + 1} = X^t + gamma * H^t (W - I) - eta * V^t
        # H^{t + 1} = H^t + c(X^{t + 1} - H^t)

        grads = self.flatten_grads(self.model.module).to(self.device) - self.flatten_grads(self.model.ref_module).to(self.device)
        self.model.flat_ref_parameters[:] = self.model.flat_parameters[:]

        self.buf.zero_()
        self.mix(self._G, self.buf)
        self.buf -= self._G
        self.V += self.gamma * self.buf + grads
        self._G += self.compression_operator(self.V - self._G)

        self.buf.zero_()
        X = self.model.flat_parameters
        self.mix(self.H, self.buf)
        self.buf -= self.H
        X += self.gamma * self.buf - self.lr * self.V
        self.H += self.compression_operator(X - self.H)

        return loss
