import torch
import torch.nn as nn

from stringcolor import cs
from vqtorch.norms import with_codebook_normalization
from .vq import VectorQuant


class ResidualVectorQuant(VectorQuant):
	"""
	Args
		groups (int): Number of residual VQ to apply. When num_residual=1, 
			layer acts will be equivalent to VectorQuant.
		share (bool): If True, codebook is shared for every quantization.
		*rest*: see VectorQuant()

	NOTE: Don't use L2 normalization on the codebook. ResidualVQ is norm sensitive.
		For norm invariant, consider using cosine distance variant.
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
			e_msg = f'feature_size {feature_size} must be divisible by residual groups {groups}.'
			raise RuntimeError(str(cs(e_msg, 'red')))

		self.groups = groups
		self.share = share

		num_codebooks = 1 if share else groups
		in_dim = self.group_size = num_codes // num_codebooks
		out_dim = feature_size

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

		z = self.prepare_inputs(z, groups=1)

		if not self.enabled:
			z = self.to_original_format(z)
			return z, {}

		######
		## (2) quantize latent vector
		######

		z_q = torch.zeros_like(z)
		z_res = torch.zeros(*z.shape[:-2], self.groups + 1, z.shape[-1]).to(z.device)

		d = torch.zeros(*z_q.shape[:-2], self.groups).to(z_q.device)
		q = torch.zeros(*z_q.shape[:-2], self.groups, dtype=torch.long).to(z_q.device)

		for i in range(self.groups):
			# select group
			_z = (z - z_q) # compute resiudal
			z_res[..., i:i+1, :] = _z

			# quantize
			cb, offset = self.get_codebook_by_group(i)
			_z_q, _d, _q = self.quantize(cb, _z)

			# update estimate
			z_q = z_q + _z_q

			# assign to tensor
			d[..., i:i+1] = _d
			q[..., i:i+1] = _q + offset
		
		z_res[..., -1:, :] = z - z_q

		to_return = {
			'z'    : z,               # each group input z_e
			'z_q'  : z_q,             # quantized output z_q
			'd'    : d,               # distance function for each group
			'q'	  : q,				  # codes using offsetted indices
			'z_res': z_res,
			'loss' : self.compute_loss(z, z_q),
			'perplexity': None,
			}

		z_q = self.straight_through_approximation(z, z_q)
		z_q = self.to_original_format(z_q)
		return z_q, to_return
