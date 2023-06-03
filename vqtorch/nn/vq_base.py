import torch
import torch.nn as nn

from vqtorch.norms import get_norm
from vqtorch.nn.utils.init import data_dependent_init_forward_hook



class _VQBaseLayer(nn.Module):
	"""
	Base template code for vector quanitzation. All VQ layers will inherit
	from this class.

	Args:
		feature_size (int):
			The size of the feature. this is the length of each
			code vector. the dimensions must match the input feature size.
		num_codes (int):
			Number of codes to use in the codebook.
		dim (int): Dimension to quantize. by default quantization happens on
			the channel dimension. For example, given an image tensor
			(B x C x H x W) and dim=1, the channels are treated as features 
			and the resulting codes `q` has the shape (B x H x W). 
			For transformers (B x N x C), you should set dim=2 or -1.
		norm (str): Feature normalization.
		codebook_norm (str): Codebook normalization.

	Returns:
		Quantized vector z_q and return dict

	Attributes:
		cdist_chunk_size (int): chunk size for divide-and-conquer topk cdist.
		enabled (bool): If false, the model is not quantized and acts as an identity layer.
	"""

	cdist_chunk_size = 1024
	enabled = True

	def __init__(
			self,
			feature_size : int,
			num_codes :	int,
			dim : int = 1,
			norm :	str = 'none',
			cb_norm : str = 'none',
			kmeans_init : bool = False,
			code_vector_size : int = None,
			):

		super().__init__()
		self.feature_size = feature_size
		self.code_vector_size = feature_size if code_vector_size is None else code_vector_size
		self.num_codes = num_codes
		self.dim = dim

		self.groups = 1 # for group VQ
		self.topk = 1   # for probabilistic VQ

		self.norm = norm
		self.codebook_norm = cb_norm
		self.norm_layer, self.norm_before_grouping = get_norm(norm, feature_size)

		if kmeans_init:
			self.register_buffer('data_initialized', torch.zeros(1))
			self.register_forward_hook(data_dependent_init_forward_hook)
		return

	def quantize(self, codebook, z):
		"""
		Quantizes the latent codes z with the codebook

		Args:
			codebook (Tensor): B x C
			z (Tensor): B x ... x C
		"""
		raise NotImplementedError


	def compute_loss(self, z_e, z_q):
		""" computes error between z and z_q """
		raise NotImplementedError


	def to_canonical_group_format(self, z, groups):
		"""
		Converts data into canonical group format

		The quantization dim is sent to the last dimension.
		The features are then resized such that C -> G x C'

		Args:
			x (Tensor): a tensor in group form [B x C x ... ]
			groups (int): number of groups
		Returns:
			x of shape [B x ... x G x C']
		"""

		z = z.moveaxis(self.dim, -1).contiguous()
		z = z.unflatten(-1, (groups, -1))
		return z


	def to_original_format(self, x):
		"""
		Merges group and permutes dimension back

		Args:
			x (Tensor): a tensor in group form [B x ... x G x C // G]
		Returns:
			merged `x` of shape [B x ... x C] (assuming dim=1)
		"""
		return x.flatten(-2, -1).moveaxis(-1, self.dim)


	def prepare_inputs(self, z, groups):
		"""
		Prepare input with normalization and group format

		Args:
			x (Tensor): a tensor in group form [B x C x ... ]
			groups (int): number of groups
		"""

		if len(z.shape) <= 1:
			e_msg = f'expected a tensor of at least 2 dimensions but found {z.size()}'
			raise ValueError(e_msg)

		if self.norm_before_grouping:
			z = self.norm_layer(z)

		z = self.to_canonical_group_format(z, groups)

		if not self.norm_before_grouping:
			z = self.norm_layer(z)

		return z


	@property
	def requires_grad(self):
		return self.codebook[0].weight.requires_grad


	def set_requires_grad(self, requires_grad):
		for codebook in self.codebook:
			codebook.weight.requires_grad = requires_grad
		return


	def extra_repr(self):
		repr = "\n".join([
			f"num_codes: {self.num_codes}",
			f"groups: {self.groups}",
			f"enabled: {self.enabled}",
		])
		return repr
