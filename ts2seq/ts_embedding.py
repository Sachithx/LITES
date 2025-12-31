import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================================
# SECTION 1: PATCH EMBEDDING FOR TIME SERIES
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert time series into overlapping patch embeddings.
    
    Args:
        patch_len: Length of each patch
        stride: Stride between patches (overlap = patch_len - stride)
        d_model: Embedding dimension
    """
    
    def __init__(self, patch_len: int = 8, stride: int = 4, d_model: int = 256):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Project patch to embedding
        self.projection = nn.Linear(patch_len, d_model)
        
        # Learnable positional encoding
        self.register_buffer('pe_initialized', torch.tensor(False))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: Time series [B, L] or [B, C, L]
        
        Returns:
            patch_embeds: [B, N_patches, d_model]
            n_patches: Number of patches
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, L]
        
        B, C, L = x.shape
        
        # Extract patches using unfold
        # [B, C, L] -> [B, C, N_patches, patch_len]
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # [B, C, N_patches, patch_len] -> [B, N_patches, C*patch_len]
        B, C, N, P = patches.shape
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(B, N, C * P)
        
        # Project to embedding space
        # [B, N_patches, C*patch_len] -> [B, N_patches, d_model]
        patch_embeds = self.projection(patches)
        
        return patch_embeds, N


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)