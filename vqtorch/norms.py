import torch
import torch.nn as nn
import torch.nn.functional as F


MAXNORM_CONSTRAINT_VALUE = 10


class Normalize(nn.Module):
    """
    Simple vector normalization module. By default, vectors are normalizes
    along the channel dimesion. Each vector associated to the spatial
    location is normalized. Used along with cosine-distance VQ layer.
    """
    def __init__(self, p=2, dim=1, eps=1e-6):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps
        return

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


def max_norm(w, p=2, dim=-1, max_norm=MAXNORM_CONSTRAINT_VALUE, eps=1e-8):
    norm = w.norm(p=p, dim=dim, keepdim=True)
    desired = torch.clamp(norm.data, max=max_norm)
    # desired = torch.clamp(norm, max=max_norm)
    return w * (desired / (norm + eps))


class MaxNormConstraint(nn.Module):
    def __init__(self, max_norm=1, p=2, dim=-1, eps=1e-8):
        super().__init__()
        self.p = p
        self.eps = eps
        self.dim = dim
        self.max_norm = max_norm

    def forward(self, x):
        return max_norm(x, self.p, self.dim, max_norm=self.max_norm)


@torch.no_grad()
def with_codebook_normalization(func):
    def wrapper(*args):
        self = args[0]
        for n, m in self.named_modules():
            if isinstance(m, nn.Embedding):
                if self.codebook_norm == 'l2':
                    m.weight.data = max_norm(m.weight.data, p=2, dim=1, eps=1e-8)
                elif self.codebook_norm == 'l2c':
                    m.weight.data = F.normalize(m.weight.data, p=2, dim=1, eps=1e-8)
        return func(*args)
    return wrapper


def get_norm(norm, num_channels=None):
    before_grouping = True
    if norm == 'l2':
        norm_layer = Normalize(p=2, dim=-1)
        before_grouping = False
    elif norm == 'l2c':
        norm_layer = MaxNormConstraint(p=2, dim=-1, max_norm=MAXNORM_CONSTRAINT_VALUE)
        before_grouping = False
    elif norm == 'bn':
        norm_layer = nn.BatchNorm2d(num_channels)
    elif norm == 'gn':
        norm_layer = GroupNorm(num_channels)
    elif norm in ['none', None]:
        norm_layer = nn.Identity()
    elif norm == 'in':
        norm_layer = nn.InstanceNorm2d(num_channels)
    else:
        raise ValueError(f'unknown norm {norm}')
    return norm_layer, before_grouping


def GroupNorm(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def match_norm(x, y, dim=-1, eps=1e-8):
    """
    matches vector norm of x to that of y
    Args:
        x (Tensor): a tensor of any shape
        y (Tensor): a tensor of the same shape as `x`.
        dim (int): dimension to match the norm over
        eps (float): epsilon to mitigate division by zero.
    Returns:
        `x` with the same norm as `y` across `dim`
    """
    assert x.shape == y.shape, \
        f'expected `x` and `y` to have the same dim but found {x.shape} vs {y.shape}'

    # move chosen dim to last dim
    x = x.moveaxis(dim, -1).contiguous()
    y = y.moveaxis(dim, -1).contiguous()
    x_shape = x.shape

    # unravel everything such that [GBHW X C]

    # print(x.shape)
    x = x.view(-1, x.size(-1))
    y = y.view(-1, y.size(-1))

    # compute norm on C
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)

    # clamp y_norm for division by 0
    x_norm = torch.clamp(x_norm, min=eps)

    # normalize (x now has same norm as y)
    x = y_norm * (x / x_norm)
    x = x.view(x_shape)
    x = x.moveaxis(-1, dim).contiguous()
    return x
