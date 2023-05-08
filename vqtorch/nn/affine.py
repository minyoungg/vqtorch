import math
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class AffineTransform(nn.Module):
	def __init__(
			self, 
			feature_size, 
			use_running_statistics=False, 
			momentum=0.1, 
			lr_scale=1,
			num_groups=1,
			):
		super().__init__()

		self.use_running_statistics = use_running_statistics
		self.num_groups = num_groups

		if use_running_statistics:
			self.momentum = momentum
			self.register_buffer('running_statistics_initialized', torch.zeros(1))
			self.register_buffer('running_ze_mean', torch.zeros(num_groups, feature_size))
			self.register_buffer('running_ze_var', torch.ones(num_groups, feature_size))

			self.register_buffer('running_c_mean', torch.zeros(num_groups, feature_size))
			self.register_buffer('running_c_var', torch.ones(num_groups, feature_size))
		else:
			self.scale = nn.parameter.Parameter(torch.zeros(num_groups, feature_size))
			self.bias = nn.parameter.Parameter(torch.zeros(num_groups, feature_size))
			self.lr_scale = lr_scale
		return

	@torch.no_grad()
	def update_running_statistics(self, z_e, c):
		# we find it helpful to often to make an under-estimation on the
		# z_e embedding statistics. Empirically we observe a slight
		# over-estimation of the statistics, causing the straight-through
		# estimation to grow indefinitely. While this is not an issue
		# for most model architecture, some model architectures that don't
		# have normalized bottlenecks, can cause it to eventually explode.
        # placing the VQ layer in certain layers of ViT exhibits this behavior


		if self.training and self.use_running_statistics:
			unbiased = False

			ze_mean = z_e.mean([0, 1]).unsqueeze(0)
			ze_var = z_e.var([0, 1], unbiased=unbiased).unsqueeze(0)

			c_mean = c.mean([0]).unsqueeze(0)
			c_var = c.var([0], unbiased=unbiased).unsqueeze(0)

			if not self.running_statistics_initialized:
				self.running_ze_mean.data.copy_(ze_mean)
				self.running_ze_var.data.copy_(ze_var)
				self.running_c_mean.data.copy_(c_mean)
				self.running_c_var.data.copy_(c_var)
				self.running_statistics_initialized.fill_(1)
			else:
				self.running_ze_mean = (self.momentum * ze_mean) + (1 - self.momentum) * self.running_ze_mean
				self.running_ze_var = (self.momentum * ze_var) + (1 - self.momentum) * self.running_ze_var
				self.running_c_mean = (self.momentum * c_mean) + (1 - self.momentum) * self.running_c_mean
				self.running_c_var = (self.momentum * c_var) + (1 - self.momentum) * self.running_c_var

		# wd = 0.9998 # 0.995
		# self.running_ze_mean = wd * self.running_ze_mean
		# self.running_ze_var = wd * self.running_ze_var
		return


	def forward(self, codebook):
		scale, bias = self.get_affine_params()
		n, c = codebook.shape
		codebook = codebook.view(self.num_groups, -1, codebook.shape[-1])
		codebook = scale * codebook + bias
		return codebook.reshape(n, c)


	def get_affine_params(self):
		if self.use_running_statistics:
			scale = (self.running_ze_var / (self.running_c_var + 1e-8)).sqrt()
			bias = - scale * self.running_c_mean + self.running_ze_mean
		else:
			scale = (1. + self.lr_scale * self.scale)
			bias = self.lr_scale * self.bias
		return scale.unsqueeze(1), bias.unsqueeze(1)
