<br>
<br>

<img src="./assets/vqtorch-logo.png"  width="30%">

---   

VQTorch is a PyTorch library for vector quantization. 

The library was developed and used for.   
- <a href="https://minyoungg.github.io/vqtorch/">`[1] Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks, Huh et al. ICML2023`</a>

## Installation
Development was done on Ubuntu with Python 3.9/3.10 using NVIDIA GPUs. Some requirements may need to be adjusted in order to run.
Some features, such as <b>half-precision cdist</b> and <b>cuda-based kmeans</b>, are only supported on CUDA devices.

First install the correct version of [cupy](https://github.com/cupy/cupy/). Make sure to install the correct version. The version refers to `CUDA Version` number when using the command `nvidia-smi`. `cupy` seem to now support ROCm drivers but this has not been tested.
```bash
# recent 12.x cuda versions
pip install cupy-cuda12x

# 11.x versions (for even older see the repo above)
pip install cup-cuda11x
```

Next, install `vqtorch`
```bash
git clone https://github.com/minyoungg/vqtorch
cd vqtorch
pip install -e .
```

## Example usage
For examples using `VectorQuant` for classification and auto-encoders check out [here](./examples/).

```python
import torch
from vqtorch.nn import VectorQuant

print('Testing VectorQuant')
# create VQ layer
vq_layer = VectorQuant(
                feature_size=32,     # feature dimension corresponding to the vectors
                num_codes=1024,      # number of codebook vectors
                beta=0.98,           # (default: 0.9) commitment trade-off
                kmeans_init=True,    # (default: False) whether to use kmeans++ init
                norm=None,           # (default: None) normalization for the input vectors
                cb_norm=None,        # (default: None) normalization for codebook vectors
                affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                sync_nu=0.2,         # (default: 0.0) codebook synchronization contribution
                replace_freq=20,     # (default: None) frequency to replace dead codes
                dim=-1,              # (default: -1) dimension to be quantized
                ).cuda()

# when `kmeans_init=True` is recommended to warm up the codebook before training
with torch.no_grad():
    z_e = torch.randn(128, 8, 8, 32).cuda()
    vq_layer(z_e)

# standard forward pass
z_e = torch.randn(128, 8, 8, 32).cuda()
z_q, vq_dict = vq_layer(z_e)

print(vq_dict.keys)
>>> dict_keys(['z', 'z_q', 'd', 'q', 'loss', 'perplexity'])
```

## Supported features
- `vqtorch.nn.GroupVectorQuant` - Vectors are quantized by first partitioning into `n` subvectors. 
- `vqtorch.nn.ResidualVectorQuant` - Vectors are first quantized and the residuals are repeatedly quantized.
- `vqtorch.nn.MaxVecPool2d` - Pools along the vector dimension by selecting the vector with the maximum norm.
- `vqtorch.nn.SoftMaxVecPool2d` - Pools along the vector dimension by the weighted average computed by softmax over the norm.
- `vqtorch.no_vq` - Disables all vector quantization layers that inherit `vqtorch.nn._VQBaseLayer`
```python
model = VQN(...)
with vqtorch.no_vq():
    out = model(x)
```

## Experimental features
- Group affine parameterization: divides the codebook into groups. The individual group is reparameterized with its own affine parameters. One can invoke it via 
```python
vq_layer = VectorQuant(..., affine_groups=8)
```
- In-place alternated optimization: in-place codebook during the forward pass. 
```python
inplace_optimizer = lambda *args, **kwargs: torch.optim.SGD(*args, **kwargs, lr=50.0, momentum=0.9)
vq_layer = VectorQuant(inplace_optimizer=inplace_optimizer)
```

## Planned features
We aim to incorporate commonly used VQ methods, including probabilistic VQ variants. 


## Citations
If the features such as `affine parameterization`, `synchronized commitment loss` or `alternating optimization` was useful, please consider citing

```bibtex
@inproceedings{huh2023improvedvqste,
  title={Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks},
  author={Huh, Minyoung and Cheung, Brian and Agrawal, Pulkit and Isola, Phillip},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```

If you found the library useful please consider citing
```bibtex
@misc{huh2023vqtorch,
  author = {Huh, Minyoung},
  title = {vqtorch: {P}y{T}orch Package for Vector Quantization},
  year = {2022},
  howpublished = {\url{https://github.com/minyoungg/vqtorch}},
}
```
