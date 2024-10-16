import torch
import torch.nn as nn 
import torch.nn.functional as F




class VisionTransformer(nn.Module):
  def __init__(self, positional_encoder, patch_encoder, embed_dim, n_layers, layer_norm_eps, num_heads):
    self.__super__(init)
    self.patch_encoder = patch_encoder
    self.positional_encoder = positional_encoder
    self.transformer_layers = nn.ModuleList([
                                            nn.TransformerEncoderLayer(d_model=embed_dim, n_heads=num_heads, dim_feedforward=4*embed_dim,
                                                                       activation="gelu", layer_norm_eps = layer_norm_eps)
                                            for _ in range(n_layers)])
    self.layer_norm = nn.LayerNorm(embedding_dim, layern_norm_eps)

  def forward(self, x):
    pass 


class PositionalEncoder(nn.Module):
  def __init__(self,embedding_dim, n_patches):
    self.positional_embeddings = nn.Embedding(embedding_dim,n_patches)
    self.register_buffer('positions',torch.arange(n_patches))

  def forward(self,x):
    encoded_positions = self.positional_embeddings(self.positions)
    return x + encoded_positions
   

class PatchEncoder(nn.Module):
  def __init__(self, patch_size, n_patches,embedding_dim):
    self.encoder = nn.conv3d()
    self.flatten = nn.Flatten()

def forward(self, x):
  
