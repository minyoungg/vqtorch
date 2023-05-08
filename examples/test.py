import torch
from vqtorch.nn import VectorQuant, GroupVectorQuant, ResidualVectorQuant


print('Testing VectorQuant')
# create VQ layer
vq_layer = VectorQuant(
                feature_size=32,     # feature dimension corresponding to the vectors
                num_codes=1024,      # number of codebook vectors
                beta=0.98,           # (default: 0.9) commitment trade-off
                kmeans_init=True,    # (default: False) whether to use kmeans++ init
                norm=None,           # (default: None) normalization for input vector
                cb_norm=None,        # (default: None) normalization for codebook vectors
                affine_lr=10.0,      # (default: 0.0) lr scale for affine parameters
                sync_nu=0.2,         # (default: 0.0) codebook syncronization contribution
                replace_freq=20,     # (default: None) frequency to replace dead codes
                dim=-1,              # (default: -1) dimension to be quantized
                ).cuda()

# when using `kmeans_init`, we can warmup the codebook
with torch.no_grad():
    z_e = torch.randn(128, 8, 8, 32).cuda()
    vq_layer(z_e)

# standard forward pass
z_e = torch.randn(128, 8, 8, 32).cuda()
z_q, vq_dict = vq_layer(z_e) # equivalent to above
assert z_e.shape == z_q.shape
err = ((z_e - z_q) ** 2).mean().item()
print(f'>>> quantization error: {err:.3f}')



print('Testing GroupVectorQuant')
# create VQ layer
vq_layer = GroupVectorQuant(
                feature_size=32,     
                num_codes=1024,      
                beta=0.98,    
                kmeans_init=True,    
                norm=None,         
                cb_norm=None,        
                affine_lr=10.0,   
                sync_nu=0.2,         
                replace_freq=20,     
                dim=-1,             
                groups=4,            # (default: 1) number of groups to divide the feature dimension
                share=False,         # (default: True) when True, same codebook is used for each group
                ).cuda()

# when using `kmeans_init`, we can warmup the codebook
with torch.no_grad():
    z_e = torch.randn(128, 8, 8, 32).cuda()
    vq_layer(z_e)

# standard forward pass
z_e = torch.randn(128, 8, 8, 32).cuda()
z_q, vq_dict = vq_layer(z_e) # equivalent to above
assert z_e.shape == z_q.shape
err = ((z_e - z_q) ** 2).mean().item()
print(f'>>> quantization error: {err:.3f}')




print('Testing ResidualVectorQuant')
# create VQ layer
vq_layer = ResidualVectorQuant(
                feature_size=32,     
                num_codes=1024,      
                beta=0.98,    
                kmeans_init=True,    
                norm=None,         
                cb_norm=None,        
                affine_lr=10.0,   
                sync_nu=0.2,         
                replace_freq=20,     
                dim=-1,             
                groups=4,            # (default: 1) number of groups to divide the feature dimension
                share=True,          # (default: True) when True, same codebook is used for each group
                ).cuda()

# when using `kmeans_init`, we can warmup the codebook
with torch.no_grad():
    z_e = torch.randn(128, 8, 8, 32).cuda()
    vq_layer(z_e)

# standard forward pass
z_e = torch.randn(128, 8, 8, 32).cuda()
z_q, vq_dict = vq_layer(z_e) # equivalent to above
assert z_e.shape == z_q.shape
err = ((z_e - z_q) ** 2).mean().item()
print(f'>>> quantization error: {err:.3f}')
