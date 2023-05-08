import torch



class ReplaceLRU():
	"""
	Attributes:
		rho (float): mutation noise
		timeout (int): number of batch it has seen
	"""
	VALID_POLICIES = ['input_random', 'input_kmeans', 'self']

	def __init__(self, rho=1e-4, timeout=100):
		assert timeout > 1
		assert rho > 0.0
		self.rho = rho
		self.timeout = timeout

		self.policy = 'input_random'
		# self.policy = 'input_kmeans'
		# self.policy = 'self'
		self.tau = 2.0

		assert self.policy in self.VALID_POLICIES
		return

	@staticmethod
	def apply(module, rho=0., timeout=100):
		""" register forward hook """
		fn = ReplaceLRU(rho, timeout)
		device = next(module.parameters()).device
		module.register_forward_hook(fn)
		module.register_buffer('_counts', timeout * torch.ones(module.num_codes))
		module._counts = module._counts.to(device)
		return fn

	def __call__(self, module, inputs, outputs):
		"""
		This function is triggered during forward pass
		recall: z_q, misc = vq(x)

		Args
			module (nn.VectorQuant)
			inputs (tuple): A tuple with 1 element
				x (Tensor)
			outputs (tuple): A tuple with 2 elements
				z_q (Tensor), misc (dict)
		"""
		if not module.training:
			return

		# count down all code by 1 and if used, reset timer to timeout value
		module._counts -= 1

		# --- computes most recent codebook usage --- #
		unique, counts = torch.unique(outputs[1]['q'], return_counts=True)
		module._counts.index_fill_(0, unique, self.timeout)

		# --- find how many needs to be replaced --- #
		# num_active = self.check_and_replace_dead_codes(module, outputs)
		inactive_indices = torch.argwhere(module._counts == 0).squeeze(-1)
		num_inactive = inactive_indices.size(0)

		if num_inactive > 0:

			if self.policy == 'self':
				# exponential distance allows more recently used codes to be even more preferable
				p = torch.zeros_like(module._counts)
				p[unique] = counts.float()
				p = p / p.sum()
				p = torch.exp(self.tau * p) - 1 # the negative 1 is to drive p=0 to stay 0

				selected_indices = torch.multinomial(p, num_inactive, replacement=True)
				selected_values = module.codebook.weight.data[selected_indices].clone()

			elif self.policy == 'input_random':
				z_e = outputs[1]['z'].flatten(0, -2)   # flatten to 2D
				z_e = z_e[torch.randperm(z_e.size(0))] # shuffle
				mult = num_inactive // z_e.size(0) + 1
				if mult > 1: # if theres not enough
					z_e = torch.cat(mult * [z_e])
				selected_values = z_e[:num_inactive]

			elif self.policy == 'input_kmeans':
				# can be extremely slow
				from torchpq.clustering import KMeans
				z_e = outputs[1]['z'].flatten(0, -2)   # flatten to 2D
				z_e = z_e[torch.randperm(z_e.size(0))] # shuffle
				kmeans = KMeans(n_clusters=num_inactive, distance='euclidean', init_mode="kmeans++")
				kmeans.fit(z_e.data.T.contiguous())
				selected_values = kmeans.centroids.T

			if self.rho > 0:
				norm = selected_values.norm(p=2, dim=-1, keepdim=True)
				noise = torch.randn_like(selected_values)
				selected_values = selected_values + self.rho * norm * noise

			# --- update dead codes with new codes --- #
			module.codebook.weight.data[inactive_indices] = selected_values
			module._counts[inactive_indices] += self.timeout

		return outputs



def lru_replacement(vq_module, rho=1e-4, timeout=100):
	"""
	Example::
		>>> vq = VectorQuant(...)
		>>> vq = lru_replacement(vq)
	"""
	ReplaceLRU.apply(vq_module, rho, timeout)
	return vq_module
