import math
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_shape(tensor, codebook):
	if len(tensor.shape) != 3:
		raise RuntimeError(f'expected 3d tensor but found {tensor.size()}')

	if tensor.size(2) != codebook.size(1):
		raise RuntimeError(
				f'expected tensor and codebook to have the same feature ' + \
				f'dimensions but found: {tensor.size()} vs {codebook.size()}'
				)
	return


def get_dist_fns(dist):
	if dist in ['euc', 'euclidean']:
		loss_fn = euclidean_distance
		dist_fn = euclidean_cdist_topk
	elif dist in ['cos', 'cosine']:
		loss_fn = cosine_distance
		dist_fn = cosine_cdist_topk
	else:
		raise ValueError(f'unknown distance method: {dist}')
	return loss_fn, dist_fn


def cosine_distance(z, z_q):
	"""
	Computes element wise euclidean of z and z_q

	NOTE: the euclidean distance is not a true euclidean distance.
	"""

	z = F.normalize(z, p=2, dim=-1)
	z_q = F.normalize(z_q, p=2, dim=-1)
	return euclidean_distance(z, z_q)


def euclidean_distance(z, z_q):
	"""
	Computes element wise euclidean of z and z_q

	NOTE: uses spatial averaging and no square root is applied. hence this is
	not a true euclidean distance but makes no difference in practice.
	"""
	if z.size() != z_q.size():
		raise RuntimeError(
					f'expected z and z_q to have the same shape but got ' + \
					f'{z.size()} vs {z_q.size()}'
					)

	z, z_q = z.reshape(z.size(0), -1), z_q.reshape(z_q.size(0), -1)
	return ((z_q - z) ** 2).mean(1) #.sqrt()


def euclidean_cdist_topk(tensor, codebook, compute_chunk_size=1024, topk=1,
						 half_precision=False):
	"""
	Compute the euclidean distance between tensor and every element in the
	codebook.

	Args:
		tensor (Tensor): a 3D tensor of shape [batch x HWG x feats].
		codebook (Tensor): a 2D tensor of shape [num_codes x feats].
		compute_chunk_size (int): the chunk size to use when computing cdist.
		topk (int): stores `topk` distance minimizers. If -1, topk is the
			same length as the codebook
		half_precision (bool): if True, matrix multiplication is computed
			using half-precision to save memory.
	Returns:
		d (Tensor): distance matrix of shape [batch x HWG x topk].
			each element is the distance of tensor[i] to every codebook.
		q (Tensor): code matrix of the same dimension as `d`. The index of the
			corresponding topk distances.

	NOTE: Compute chunk only looks at tensor since optimal codebook size
	generally does not vary too much. In future versions, should consider
	computing chunk size while taking into consideration of codebook and
	feature dimension size.
	"""
	check_shape(tensor, codebook)

	b, n, c = tensor.shape
	tensor_dtype = tensor.dtype
	tensor = tensor.reshape(-1, tensor.size(-1))
	tensor = tensor.split(compute_chunk_size)
	dq = []

	if topk == -1:
		topk = codebook.size(0)

	for i, tc in enumerate(tensor):
		cb = codebook

		if half_precision:
			tc = tc.half()
			cb = cb.half()

		d = torch.cdist(tc, cb)
		dq.append(torch.topk(d, k=topk, largest=False, dim=-1))

	d, q = torch.cat([_dq[0] for _dq in dq]), torch.cat([_dq[1] for _dq in dq])

	return_dict = {'d': d.to(tensor_dtype).reshape(b, n, -1),
				   'q': q.long().reshape(b, n, -1)}
	return return_dict


def cosine_cdist_topk(tensor, codebook, chunks=4, topk=1, half_precision=False):
	""" Computes cosine distance instead. see `euclidean_cdist_topk` """
	check_shape(tensor, codebook, mask)

	tensor   = F.normalize(tensor,   p=2, dim=-1)
	codebook = F.normalize(codebook, p=2, dim=-1)

	d, q = euclidean_cdist_topk(tensor, codebook, chunks, topk, half_precision)

	d = 0.5 * (d ** 2)
	return d, q.long()
