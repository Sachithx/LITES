import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from ts2seq.ts_embedding import PatchEmbedding, PositionalEncoding

# ============================================================================
# SECTION 3: T5-STYLE DECODER WITH CROSS-ATTENTION
# ============================================================================

class EventTextDecoder(nn.Module):
    """
    T5-style decoder for generating event text autoregressively.
    
    Args:
        vocab_size: Size of text vocabulary
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of decoder layers
        dim_feedforward: FFN dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target tokens [B, T]
            memory: Encoder outputs [B, N, d_model]
            tgt_mask: Causal mask [T, T]
            tgt_key_padding_mask: Padding mask [B, T]
        
        Returns:
            logits: [B, T, vocab_size]
        """
        # Embed tokens
        tgt_embed = self.token_embed(tgt) * math.sqrt(self.d_model)  # [B, T, d_model]
        
        # Add positional encoding
        tgt_embed = self.pos_encoder(tgt_embed)
        tgt_embed = self.dropout(tgt_embed)
        
        # Decoder with cross-attention
        decoded = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # [B, T, d_model]
        
        # Project to vocabulary
        logits = self.output_proj(decoded)  # [B, T, vocab_size]
        
        return logits
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

