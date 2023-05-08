

def entropy(x, dim=-1, eps=1e-8, keepdim=False):
	assert x.min() >= 0., \
		f'function takes non-negative values but found x.min(): {x.min()}'
	is_tensor = True

	if len(x.shape) == 1:
		is_tensor = False
		x = x.unsqueeze(0)

	x = x.moveaxis(dim, -1)
	x_shape = x.shape
	x = x.view(-1, x.size(-1)) + eps
	p = x / x.sum(dim=1, keepdim=True)
	h = - (p * p.log()).sum(dim=1, keepdim=True)
	h = h.view(*x_shape[:-1], 1).moveaxis(dim, -1)

	if not keepdim:
		h = h.squeeze(dim)

	if not is_tensor:
		h = h.squeeze(0)
	return h
