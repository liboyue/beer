import torch
from beer.optim import CompressedOptimizer
from beer.utils import flatten_tensors

from beer import log

class CHOCO_SGD(CompressedOptimizer):
    def __init__(self, model, gamma=0.1, **kwargs):
        super().__init__(model, **kwargs)
        self.gamma = gamma
        self.device = next(model.module.parameters()).device

        self.buf = torch.zeros_like(model.flat_parameters, device=self.device)
        self.x_hat = torch.zeros_like(model.flat_parameters, device=self.device)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Original CHOCO-SGD
        # X^t = X^{t - 0.5} + gamma * X_hat^t (W - I)
        # Q^t = compress(X^t - X_hat^t)
        # X_hat^{t+1} = X_hat^t + Q^t
        # X^{t + 0.5} = X^t - eta * g^t

        # Re-order
        # X^t = X^{t-1} - eta * g^t + gamma * X_hat^t (W - I)
        # Q^t = compress(X^t - X_hat^t)
        # X_hat^{t+1} = X_hat^t + Q^t

        x = self.model.flat_parameters

        grads = flatten_tensors([t.grad for t in self.model.module.parameters()]).to(self.device)
        x -= self.lr * grads

        self.x_hat += self.compression_operator(x - self.x_hat)

        self.buf.zero_()
        self.mix(self.x_hat, self.buf)
        x += self.gamma * (self.buf - self.x_hat)

        return loss
