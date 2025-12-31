import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from ts2seq.ts_encoder import TimeSeriesEncoder
from ts2seq.text_decoder import EventTextDecoder

# ============================================================================
# SECTION 4: COMPLETE TIME SERIES TO TEXT MODEL
# ============================================================================

class TimeSeriesEventModel(nn.Module):
    """
    Complete encoder-decoder model for time series event text generation.
    
    Architecture:
        Encoder: Time series → Patch embeddings → Transformer
        Decoder: Text tokens + Cross-attention → Autoregressive text
    
    Args:
        vocab_size: Size of text vocabulary
        d_model: Model dimension (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Encoder depth (default: 6)
        num_decoder_layers: Decoder depth (default: 6)
        dim_feedforward: FFN dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Encoder: Time series → representations
        self.encoder = TimeSeriesEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Decoder: Text generation with cross-attention
        self.decoder = EventTextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        
        Args:
            src: Time series [B, L]
            tgt: Target tokens [B, T] (shifted right)
            tgt_padding_mask: Padding mask [B, T]
        
        Returns:
            logits: [B, T, vocab_size]
        """
        # Encode time series
        memory = self.encoder(src)  # [B, N_patches, d_model]
        
        # Generate causal mask
        T = tgt.size(1)
        tgt_mask = self.decoder.generate_square_subsequent_mask(
            T, device=tgt.device
        )
        
        # Decode with cross-attention
        logits = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_length: int = 100,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> torch.Tensor:
        """
        Autoregressive generation (greedy decoding).
        
        Args:
            src: Time series [B, L]
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            temperature: Sampling temperature
            top_k: Top-k sampling
        
        Returns:
            generated: [B, max_length]
        """
        self.eval()
        
        B = src.size(0)
        device = src.device
        
        # Encode time series once
        memory = self.encoder(src)  # [B, N, d_model]
        
        # Start with BOS token
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Generate causal mask
            T = generated.size(1)
            tgt_mask = self.decoder.generate_square_subsequent_mask(T, device)
            
            # Forward pass
            logits = self.decoder(
                tgt=generated,
                memory=memory,
                tgt_mask=tgt_mask
            )  # [B, T, vocab_size]
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = torch.gather(
                    top_k_indices, 
                    1, 
                    torch.multinomial(probs, 1)
                )
            else:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences finished
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

