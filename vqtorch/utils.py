import numpy as np
import torch

from vqtorch.nn import _VQBaseLayer


def is_vq(m):
	""" checks if the module is a VQ layer """
	return issubclass(type(m), _VQBaseLayer)

def is_vqn(net):
	""" checks if the network contains a VQ layer """
	return np.any([is_vq(layer) for layer in net.modules()])

def get_vq_layers(model):
	""" returns vq layers from a network """
	return [m for m in model.modules() if is_vq(m)]


class no_vq():
    """
    Function to turn off VQ by setting all the VQ layers to identity function.

    Examples::
        >>> with vqtorch.no_vq():
        ...     out = model(x)
    """

    def __init__(self, modules):
        if type(modules) is not list:
            modules = [modules]

        for module in modules:
            if isinstance(module, torch.nn.DataParallel):
                module = module.module

            assert isinstance(module, torch.nn.Module), \
                f'expected input to be nn.Module or a list of nn.Module ' + \
                f'but found {type(module)}'

            self.enable_vq(module, enable=False)

        self.modules = modules
        return

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        for module in self.modules:
            if isinstance(module, torch.nn.DataParallel):
                module = module.module
            self.enable_vq(module, enable=True)
        return

    def enable_vq(self, module, enable):
        for m in module.modules():
            if is_vq(m):
                m.enabled = enable
        return
