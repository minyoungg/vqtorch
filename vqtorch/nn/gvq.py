import torch
import torch.nn as nn

from stringcolor import cs
from vqtorch.norms import with_codebook_normalization
from .vq import VectorQuant


class GroupVectorQuant(VectorQuant):
	"""
	Vector quantization layer.

	Args:
		groups (int): Number of groups for vector quantization. The vectors are divided
			into group chunks. When groups=1, it is the same as VectorQuant.
		share (bool): If True, codebook is shared for each sub-vector.
		*rest*: see VectorQuant()
	"""

	def __init__(
			self,
			feature_size : int,
			num_codes : int,
			groups : int = 1,
			share : bool = True,
			**kwargs,
			):

		if not share and not feature_size % groups == 0:
			e_msg = f'feature_size {self.feature_size} must be divisible by groups {groups}.'
			raise RuntimeError(str(cs(e_msg, 'red')))

		num_codebooks = 1 if share else groups
		in_dim  = self.group_size = num_codes // num_codebooks
		out_dim = feature_size // groups

		super().__init__(feature_size, num_codes, code_vector_size=out_dim, **kwargs)

		self.groups = groups
		self.share = share
		self.codebook = nn.Embedding(in_dim * num_codebooks, out_dim)
		return
	

	def get_codebook_by_group(self, group):
		cb = self.codebook.weight
		offset = 0 if self.share else group * self.group_size
		return cb[offset : offset + self.group_size], offset


	@with_codebook_normalization
	def forward(self, z):

		######
		## (1) formatting data by groups and invariant to dim
		######

		z = self.prepare_inputs(z, self.groups)

		if not self.enabled:
			z = self.to_original_format(z)
			return z, {}

		######
		## (2) quantize latent vector
		######

		z_q = torch.zeros_like(z)
		d = torch.zeros(z_q.shape[:-1]).to(z_q.device)
		q = torch.zeros(z_q.shape[:-1], dtype=torch.long).to(z_q.device)

		for i in range(self.groups):
			# select group
			_z = z[..., i:i+1, :]

			# quantize
			cb, offset = self.get_codebook_by_group(i)
			_z_q, _d, _q = self.quantize(cb, _z)

			# assign to tensor
			z_q[..., i:i+1, :] = _z_q
			d[..., i:i+1] = _d
			q[..., i:i+1] = _q + offset

		to_return = {
			'z'   : z,               # each group input z_e
			'z_q' : z_q,             # quantized output z_q
			'd'   : d,               # distance function for each group
			'q'	  : q,				 # codes using offsetted indices
			'loss': self.compute_loss(z, z_q),
			'perplexity': None,
			}

		z_q = self.straight_through_approximation(z, z_q)
		z_q = self.to_original_format(z_q)
		return z_q, to_return
