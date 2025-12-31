import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from ts2seq.ts_embedding import PatchEmbedding, PositionalEncoding

# ============================================================================
# SECTION 2: TRANSFORMER ENCODER
# ============================================================================

class TimeSeriesEncoder(nn.Module):
    """
    Transformer encoder for time series representations.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: FFN dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_len=8,
            stride=4,
            d_model=d_model
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Time series [B, L]
        
        Returns:
            encoded: [B, N_patches, d_model]
        """
        # Patch embedding
        patch_embeds, n_patches = self.patch_embed(x)  # [B, N, d_model]
        
        # Add positional encoding
        patch_embeds = self.pos_encoder(patch_embeds)
        patch_embeds = self.dropout(patch_embeds)
        
        # Transformer encoding
        encoded = self.transformer_encoder(patch_embeds)  # [B, N, d_model]
        
        return encoded
