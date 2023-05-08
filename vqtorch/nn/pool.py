import torch
import torch.nn as nn
import torch.nn.functional as F


class _VecPool2d(nn.Module):
    def __init__(self, weighting_fn, kernel_size=3, stride=2, padding=2, dilation=1):
        super().__init__()

        self.k = kernel_size
        self.s = stride
        self.p = padding
        self.d = dilation

        self.weighting_fn = weighting_fn

        self.unfold = nn.Unfold(kernel_size=self.k, dilation=self.d, padding=self.p, stride=self.s)
        return

    def forward(self, x):
        b, c, h, w = x.shape
        out_h = (h - self.k + 2 * self.p) // self.s + 1
        out_w = (w - self.k + 2 * self.p) // self.s + 1

        with torch.no_grad():
            n = x.norm(dim=1, p=2, keepdim=True)
            n = self.unfold(n) # B x (K**2) x N
            n = self.weighting_fn(n, dim=1).unsqueeze(1)

        x = self.unfold(x)  # B x C * (K**2) x N
        x = x.view(b, c, -1, x.size(-1))

        x = n * x
        x = x.sum(2).view(b, c, out_h, out_w)
        return x


class MaxVecPool2d(_VecPool2d):
    def __init__(self, *args, **kwargs):
        super().__init__(MaxVecPool2d.max_onehot, *args, **kwargs)

    @staticmethod
    def max_onehot(x, dim):
        b, s, n = x.shape

        x = x.argmax(dim=1)
        x = x.view(-1)
        x = F.one_hot(x, s).view(b, n, s).swapaxes(-1, dim)
        return x


class SoftMaxVecPool2d(_VecPool2d):
    def __init__(self, *args, **kwargs):
        super().__init__(torch.softmax, *args, **kwargs)
