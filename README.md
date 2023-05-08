# VQTorch

## Installation

```bash
git clone https://github.com/minyoungg/vqtorch
cd vqtorch
pip install -e .
```
## 

Some design choices are sub-optimal so I am planning to refactor them eventually.
Here are some examples to get you started.
Things to note before getting started.
- minimum dimension supported is 3 dimensions. (will be fixed later). If you want to quantize a matrix, unsqueeze the first dimension and squeeze it back.
- The commitment loss is computed inside the layer, this may be refactored to be computed outside.
- Has to use GPUs because half-precision `cdist` is not supported on CPU.

```python
from vqtorch import VectorQuantize
```

#### Euclidean VQ (original)
This uses Euclidean distance, which is the default distance argument

```python
vq = VectorQuantize(feature_size=256, num_codes=512)
```

#### Cosine VQ
This uses cosine distance instead of euclidean distance. It's significantly more stable and easier to train with.

By default the norm of code-vectors are unit-normalized and the input norm is not preserved.

<b>(recommended starting point)</b>
```python 
vq = VectorQuantize(feature_size=256, num_codes=512, dist='cosine')
```

You can set the following argument to preserve the norm. This one tends to be slightly more less stable.
```python
vq = VectorQuantize(feature_size=256, num_codes=512, dist='cosine', vector_normalize=True)
```

#### Grouped VQ
Grouped VQ is the idea of breaking up a vector into a group of sub-vectors and quantizing them independently.
This allows the quantization to be able to better precisely model the input tensor and improve the training of the codebook by increasing the selection rate.
While using more groups generally improves performance, the caveat is that the nice properties of VQ may disappear.

```python 
vq = VectorQuantize(feature_size=256, num_codes=512, dist='cosine', groups=4)
```

---
## How to use it
Use it like any other PyTorch layer

```python
from vqtorch import VectorQuantize
vq = VectorQuantize(feature_size=32, num_codes=256).cuda()

# input tensor
z_e = torch.randn(1, 32, 64, 64).cuda()

# quantized tensor
z_q, misc = vq(z_e)

print(z_q.shape)
>>> torch.Size([1, 32, 64, 64])

# commitment loss (scaled by the beta hyper-parameters in the layer)
commitment_loss = misc['loss']
```
---
## Advanced 

#### Quantization dimension
By default, we quantize along the `dim=1` which is the channel dimension for image Tensors `BCHW`.  
If you want to quantize along a different dimension, you can specify it in the function.  
For example, for Transformers with input dimensions `BNF` where `N` is the number of tokens and `F` is the features.
One would set the layers to be:

```python
vq = VectorQuantize(feature_size=256, num_codes=512, dim=2)
```

#### Warming up codebook (important)
The codebook initialization is extremely important. A badly initialized codebook, the default one, will make it more or less untrainable.
Hence, you want to initialize your codebook from the data instead. I provide a function to help you with this

```python
import ... #(all your important libraries)
from vqtorch.inits import codebook_warmup

class MyVQNetwork(nn.Module):
  def __init__(self):
    ... # your sophisticated model with VectorQuantize layers in it
    
# initialize your network
model = MyVQNetwork()

# initialize your data
train_loader, test_loader = get_data(...)

# warms up the codebook using `num_batch` forward passes
codebook_warmup(model, train_loader, warmup_method='data', num_batch=128)

# Your usual training loop
train(model, ...)
```

(Under the hood) the `codebook_warmup` applies a hook to all VectorQuantize layers and is triggered in the forward pass to replace the code-vectors in the codebook at random with the input data.
