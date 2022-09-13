import torch
from torch.optim.optimizer import Optimizer, required
import torch.functional as F


class NormGrad(Optimizer):
    def __init__(self, params, lr=required, beta=required):
        defaults = {lr: lr,
                    beta: beta}

        super(NormGrad, self).__init__(params, default=defaults)

    def __setstate__(self, state):
        super(NormGrad, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for groups in self.param_groups:
            beta = groups['beta']
            lr = groups['lr']
            device = groups['params'].data.device

            for p in groups['params']:

                if len(p.grad.shape) == 2:
                    x = p.grad
                    y = torch.norm(x, dim=1)
                    x = F.normalize(x, dim=1)
                    y = (torch.tile(y.permute(1, 0), dims=(p.grad.shape[1], 1))).reshape(p.grad.shape[0], p.grad.shape[1])
                else:
                    y = p.grad
                    x = torch.ones(device=device, size=p.grad.shape, dtype=torch.double)

                y[beta < y] = beta
                x = y * x

                p.data.add_(x, alpha=-lr)

        return loss

