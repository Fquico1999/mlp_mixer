import einops
import torch.nn as nn

class MLPBlock(nn.Module):
  """
  Parameters:
  dim: int - Input and output dimension of the entire block. One of n_patches or hidden_dim.
  mlp_dim: int - Dimension of the hidden layer
  """
  def __init__(self, dim, mlp_dim=None):
    super().__init__()

    # Check if mlp_dim is provided 
    mlp_dim = dim if mlp_dim is None else mlp_dim
    self.fc1 = nn.Linear(dim, mlp_dim)
    self.gelu = nn.GELU()
    self.fc2 = nn.Linear(mlp_dim, dim)

  def forward(self, x):
    """
    Parameters:
    x: Tensor - Input of shape [samples, channels, patches] for token mixing
                or [samples, patches, channels] for channel mixing.

    Returns:
    Tensor - Output tensor with same shape as x
    """
    x = self.fc1(x)   # [samples, *, mlp_dim]
    x = self.gelu(x)  # [samples, *, mlp_dim]
    x = self.fc2(x)   # [samples, *, dim]
    return x


class MixerBlock(nn.Module):
  """
  Mixer Block that contains two MLPBlocks and layer norms

  Parameters:
  n_patches: int - Number of image patches
  hidden_dim: int - Dimension of patch embeddings
  tokens_mlp_dim: int - Hidden dimension for token mixong MLPBlock
  channel_mlp_dim: int - Hidden dimension for channel mixing MLPBlock
  """
  def __init__(self, *, n_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
    super().__init__()
    # Normalization along patch embedding dimension
    self.norm_1 = nn.LayerNorm(hidden_dim)
    self.norm_2 = nn.LayerNorm(hidden_dim)

    self.token_mixing_block = MLPBlock(n_patches, tokens_mlp_dim)
    self.channel_mixing_block = MLPBlock(hidden_dim, channels_mlp_dim)


  def forward(self, x):
    """
    Parameters:
    x: Tensor - Input tensor of shape [samples, n_patches, hidden_dim]

    Returns:
    Tensor - Output tensor of equal shape as x.
    """

    y = self.norm_1(x)      # [samples, patches, hidden_dim]
    # For token mixing, require last dimension to be patches
    y = y.permute(0, 2, 1)  # [samples, hidden_dim, patches]
    y = self.token_mixing_block(y)  # [samples, hidden_dim, patches]
    # Undo permutation for skip connection
    y = y.permute(0, 2, 1)  # [samples, patches, hidden_dim]
    x = x + y   # [samples, patches, hidden_dim]

    y = self.norm_2(x)  # [samples, patches, hidden_dim]
    y = self.channel_mixing_block(y)  # [samples, patches, hidden_dim]
    return x + y  # [samples, patches, hidden_dim]


class MLPMixer(nn.Module):
  """
  MLPMixer Network.

  Parameters:
  img_size: int - Size of input image H=W
  patch_size: int - Height and width of square patches. 
  tokens_mlp_dim: int - Hidden dimension of token mixing MLPBlock
  channels_mlp_dim: int - Hidden dimension of channel mixing MLPBlock
  n_classes: int - Number of classes used in classification
  hidden_dim: int - Patch embedding dimension
  n_blocks: int - Number of MixerBlocks. a.k.a depth.
  """
  def __init__(self, *, img_size, patch_size, tokens_mlp_dim, channels_mlp_dim, n_classes, hidden_dim, n_blocks):
    super().__init__()
    # Derive number of patches from img_size
    assert img_size % patch_size == 0, "Require img_size to be divisible by patch_size"
    n_patches = (img_size // patch_size) ** 2
    self.n_classes = n_classes
    
    # Embed patches from original image using 2D convolution with stride=patch_size
    self.patch_embedder = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
    
    # Initialize n_block MixerBlocks
    self.blocks = nn.ModuleList([MixerBlock(n_patches=n_patches, 
                                            hidden_dim=hidden_dim, 
                                            tokens_mlp_dim=tokens_mlp_dim, 
                                            channels_mlp_dim=channels_mlp_dim)
                                  for i in range(n_blocks)])
    
    # Pre classification normalization
    self.norm = nn.LayerNorm(hidden_dim)
    self.classification_head = nn.Linear(hidden_dim, n_classes)


  def forward(self, x, y=None):
    """
    Parameters:
    x: Tensor - Input batch of images of shape [samples, channels, img_size, img_size]
    y: Tensor - Labels for training to return loss
    Returns:
    Tensor - Class logits of shape [samples, n_classes]
    """
    # Extract and embed patches for each image
    x = self.patch_embedder(x)  # [samples, hidden_dim, sqrt(n_patches), sqrt(n_patches)]
    
    # To go into mixer blocks we need to collapse patches into one dimension and swap channels
    x = einops.rearrange(x, "n c h w -> n (h w) c") # [samples, n_patches, hidden_dim]

    # Apply n_block MixerBlocks
    for block in self.blocks:
      x = block(x)

    # Apply Global Average Pooling
    x = self.norm(x)  # [samples, n_patches, hidden_dim]
    x = x.mean(dim=1) # [samples, hidden_dim]

    # Apply classification head
    logits = self.classification_head(x) # [samples, n_classes]
    
    if y is not None:
        loss_fnc = nn.CrossEntropyLoss()
        loss = loss_fnc(logits.view(-1, self.n_classes), y.view(-1))
        return loss
    else:
        return logits