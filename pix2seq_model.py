"""
PIX2SEQ ARCHITECTURE ADAPTED FOR TIME SERIES
============================================

PyTorch implementation of Pix2Seq encoder-decoder architecture for time series
interval detection. This replaces image patches with 1D time series windows while
keeping the autoregressive decoder unchanged.

Key Adaptations:
    - 1D Conv encoder instead of 2D image patches
    - Temporal positional embeddings instead of spatial
    - Output format: [class, x_min, x_max] instead of [class, x_min, y_min, x_max, y_max]
    - Works with hierarchical event intervals

Author: Sachith Abeywickrama
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TimeSeriesPix2SeqConfig:
    """Configuration for time series Pix2Seq model."""
    
    # Time series input
    seq_len: int = 128                    # Length of input time series
    input_features: int = 1               # Number of input features
    window_size: int = 8                  # Window size for tokenization
    
    # Encoder
    num_encoder_layers: int = 2
    dim_att: int = 32                    # Attention dimension
    dim_mlp: int = 128                   # MLP dimension
    num_heads: int = 2
    encoder_dropout: float = 0.1
    encoder_drop_path: float = 0.1
    
    # Decoder
    num_decoder_layers: int = 2
    dim_att_dec: int = 64                # Decoder attention dimension
    dim_mlp_dec: int = 256               # Decoder MLP dimension
    num_heads_dec: int = 2
    decoder_dropout: float = 0.1
    decoder_drop_path: float = 0.1
    shared_decoder_embedding: bool = True
    
    # Vocabulary and sequence
    num_classes: int = 64                 # Number of event classes (from hierarchical vocab)
    quantization_bins: int = 1000         # Discretization resolution for positions
    max_seq_len: int = 512                # Max sequence length (max_intervals * 3)
    max_intervals: int = 100              # Maximum intervals per sample
    
    # Special tokens
    pad_token: int = 0
    eos_token: int = 1
    class_vocab_start: int = 100          # Classes start at 100
    coord_vocab_start: int = 1100         # Coordinates start at 1100
    
    # Positional encoding
    pos_encoding: str = 'learned'         # 'learned' or 'sincos'
    pos_encoding_dec: str = 'learned'
    
    # Training
    label_smoothing: float = 0.1
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.4
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        # Special (100) + Classes (num_classes) + Coordinates (quantization_bins)
        return 100 + self.num_classes + self.quantization_bins
    
    @property
    def num_windows(self) -> int:
        """Number of windows after tokenization."""
        return self.seq_len // self.window_size


# ============================================================================
# POSITIONAL ENCODINGS
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, non-learnable)."""
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D] with added positional encoding
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""
    
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D] with added positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


# ============================================================================
# TRANSFORMER COMPONENTS
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: [B, L, D]
            src_mask: Optional attention mask
        Returns:
            [B, L, D]
        """
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: [B, L_tgt, D]
            memory: [B, L_mem, D]
            tgt_mask: Causal mask for self-attention
            memory_mask: Optional memory mask
        Returns:
            [B, L_tgt, D]
        """
        # Self-attention (causal)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention to encoder output
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


# ============================================================================
# TIME SERIES ENCODER
# ============================================================================

class TimeSeriesEncoder(nn.Module):
    """
    1D Time Series Encoder using windowed convolutions.
    
    Replaces 2D image patches with 1D time series windows.
    Architecture matches Pix2Seq but adapted for 1D.
    """
    
    def __init__(self, config: TimeSeriesPix2SeqConfig):
        super().__init__()
        self.config = config
        
        # 1D convolution as "stem" (replaces 2D patch embedding)
        self.stem_conv = nn.Conv1d(
            in_channels=config.input_features,
            out_channels=config.dim_att,
            kernel_size=config.window_size,
            stride=config.window_size,  # Non-overlapping windows
            padding=0
        )
        
        self.stem_ln = nn.LayerNorm(config.dim_att)
        
        # Positional embeddings
        if config.pos_encoding == 'learned':
            self.pos_emb = LearnedPositionalEncoding(
                config.num_windows, config.dim_att
            )
        else:
            self.pos_emb = SinusoidalPositionalEncoding(
                config.num_windows, config.dim_att
            )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.dim_att,
                config.num_heads,
                config.dim_mlp,
                config.encoder_dropout
            )
            for _ in range(config.num_encoder_layers)
        ])
        
        self.output_ln = nn.LayerNorm(config.dim_att)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode time series into latent representations.
        
        Args:
            x: Time series [B, L, F] where L=seq_len, F=input_features
            mask: Optional mask [B, L]
        
        Returns:
            Encoded tokens [B, num_windows, dim_att]
        """
        B, L, F = x.shape
        
        # Transpose for Conv1d: [B, F, L]
        x = x.transpose(1, 2)
        
        # Apply 1D convolution to create windows/tokens
        tokens = self.stem_conv(x)  # [B, dim_att, num_windows]
        tokens = tokens.transpose(1, 2)  # [B, num_windows, dim_att]
        
        # Layer norm
        tokens = self.stem_ln(tokens)
        
        # Add positional embeddings
        tokens = self.pos_emb(tokens)
        
        # Transformer encoding
        for layer in self.encoder_layers:
            tokens = layer(tokens, src_mask=mask)
        
        # Final layer norm
        tokens = self.output_ln(tokens)
        
        return tokens


# ============================================================================
# AUTOREGRESSIVE DECODER
# ============================================================================

class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive decoder for sequence generation.
    
    This is nearly identical to Pix2Seq decoder - only the output
    vocabulary changes (no y coordinates).
    """
    
    def __init__(self, config: TimeSeriesPix2SeqConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim_att_dec)
        
        # Positional embeddings
        if config.pos_encoding_dec == 'learned':
            self.pos_emb = LearnedPositionalEncoding(
                config.max_seq_len, config.dim_att_dec
            )
        else:
            self.pos_emb = SinusoidalPositionalEncoding(
                config.max_seq_len, config.dim_att_dec
            )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                config.dim_att_dec,
                config.num_heads_dec,
                config.dim_mlp_dec,
                config.decoder_dropout
            )
            for _ in range(config.num_decoder_layers)
        ])
        
        # Output projection
        if config.shared_decoder_embedding:
            # Share weights with input embedding
            self.output_proj = nn.Linear(config.dim_att_dec, config.vocab_size, bias=False)
            self.output_proj.weight = self.token_embedding.weight
        else:
            self.output_proj = nn.Linear(config.dim_att_dec, config.vocab_size)
        
        self.output_ln = nn.LayerNorm(config.dim_att_dec)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, seq: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            seq: Target sequence tokens [B, L]
            memory: Encoded time series [B, num_windows, dim_att]
            tgt_mask: Optional target mask
        
        Returns:
            Logits [B, L, vocab_size]
        """
        B, L = seq.shape
        device = seq.device
        
        # Embed tokens
        x = self.token_embedding(seq)  # [B, L, dim_att_dec]
        
        # Add positional embeddings
        x = self.pos_emb(x)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(L, device)
        
        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask)
        
        # Output layer norm
        x = self.output_ln(x)
        
        # Project to vocabulary
        logits = self.output_proj(x)  # [B, L, vocab_size]
        
        return logits
    
    @torch.no_grad()
    def generate(self, memory: torch.Tensor, max_length: int,
                 temperature: float = 1.0, top_k: int = 0,
                 top_p: float = 0.0, eos_token: int = 1) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            memory: Encoded time series [B, num_windows, dim_att]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (0 = disabled)
            eos_token: End-of-sequence token ID
        
        Returns:
            Generated sequence [B, L]
        """
        B = memory.size(0)
        device = memory.device
        
        # Start with empty sequence
        generated = torch.zeros((B, 1), dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for step in range(max_length):
            # Get logits for current sequence
            logits = self.forward(generated, memory)  # [B, step+1, vocab_size]
            next_token_logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # Mark finished sequences
            finished |= (next_token.squeeze(-1) == eos_token)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences finished
            if finished.all():
                break
        
        return generated


# ============================================================================
# PROJECTION LAYER
# ============================================================================

class EncoderToDecoderProjection(nn.Module):
    """Project encoder output to decoder dimension."""
    
    def __init__(self, dim_enc: int, dim_dec: int):
        super().__init__()
        self.linear = nn.Linear(dim_enc, dim_dec)
        self.ln = nn.LayerNorm(dim_dec)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, dim_enc]
        Returns:
            [B, L, dim_dec]
        """
        x = self.linear(x)
        x = self.ln(x)
        return x


# ============================================================================
# COMPLETE MODEL
# ============================================================================

class TimeSeriesPix2Seq(nn.Module):
    """
    Complete Pix2Seq model adapted for time series interval detection.
    
    Architecture:
        1. Time Series Encoder (1D Conv + Transformer)
        2. Projection Layer
        3. Autoregressive Decoder (Transformer)
    
    Input:  Time series [B, L, F]
    Output: Interval sequences [class, x_min, x_max, class, x_min, x_max, ...]
    """
    
    def __init__(self, config: TimeSeriesPix2SeqConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = TimeSeriesEncoder(config)
        
        # Projection (if encoder and decoder dims differ)
        if config.dim_att != config.dim_att_dec:
            self.projection = EncoderToDecoderProjection(
                config.dim_att, config.dim_att_dec
            )
        else:
            self.projection = nn.Identity()
        
        # Decoder
        self.decoder = AutoregressiveDecoder(config)
    
    def encode(self, timeseries: torch.Tensor) -> torch.Tensor:
        """
        Encode time series.
        
        Args:
            timeseries: [B, L, F]
        
        Returns:
            Encoded representation [B, num_windows, dim_att_dec]
        """
        encoded = self.encoder(timeseries)
        encoded = self.projection(encoded)
        return encoded
    
    def forward(self, timeseries: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            timeseries: [B, L, F] - Input time series
            seq: [B, L_seq] - Target sequence tokens
        
        Returns:
            Logits [B, L_seq, vocab_size]
        """
        # Encode time series
        encoded = self.encode(timeseries)
        
        # Decode to sequence
        logits = self.decoder(seq, encoded)
        
        return logits
    
    @torch.no_grad()
    def predict(self, timeseries: torch.Tensor, max_intervals: Optional[int] = None,
                temperature: float = 1.0, top_k: int = 0, top_p: float = 0.4) -> torch.Tensor:
        """
        Generate interval predictions.
        
        Args:
            timeseries: [B, L, F]
            max_intervals: Maximum number of intervals to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            Generated sequences [B, L_gen]
        """
        if max_intervals is None:
            max_intervals = self.config.max_intervals
        
        # Encode
        encoded = self.encode(timeseries)
        
        # Generate
        max_length = max_intervals * 3 + 1  # class + x_min + x_max per interval + EOS
        generated = self.decoder.generate(
            encoded,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token=self.config.eos_token
        )
        
        return generated
    
    def decode_sequence(self, sequence: torch.Tensor) -> List[Dict]:
        """
        Convert predicted sequence tokens back to intervals.
        
        Args:
            sequence: [L] - Token sequence
        
        Returns:
            List of interval dicts with keys: class_id, x_min, x_max
        """
        intervals = []
        
        i = 0
        while i < len(sequence):
            token = sequence[i].item()
            
            # Check for EOS
            if token == self.config.eos_token:
                break
            
            # Check if we have enough tokens for an interval
            if i + 2 >= len(sequence):
                break
            
            # Decode class
            class_token = sequence[i].item()
            if class_token < self.config.class_vocab_start:
                i += 1
                continue
            class_id = class_token - self.config.class_vocab_start
            
            # Decode positions
            x_min_token = sequence[i + 1].item()
            x_max_token = sequence[i + 2].item()
            
            x_min = (x_min_token - self.config.coord_vocab_start) / self.config.quantization_bins
            x_max = (x_max_token - self.config.coord_vocab_start) / self.config.quantization_bins
            
            # Validate
            if 0.0 <= x_min < x_max <= 1.0:
                intervals.append({
                    'class_id': class_id,
                    'x_min': x_min,
                    'x_max': x_max
                })
            
            i += 3
        
        return intervals


# ============================================================================
# SEQUENCE TOKENIZATION UTILITIES
# ============================================================================

def intervals_to_sequence(intervals: List[Dict], config: TimeSeriesPix2SeqConfig) -> torch.Tensor:
    """
    Convert interval labels to token sequence.
    
    Args:
        intervals: List of dicts with keys: class_id, x_min, x_max
        config: Model configuration
    
    Returns:
        Token sequence [L]
    """
    sequence = []
    
    for interval in intervals:
        # Class token
        class_token = interval['class_id'] + config.class_vocab_start
        
        # Quantize positions to discrete tokens
        x_min_token = int(interval['x_min'] * config.quantization_bins) + config.coord_vocab_start
        x_max_token = int(interval['x_max'] * config.quantization_bins) + config.coord_vocab_start
        
        # Clamp to valid range
        x_min_token = max(config.coord_vocab_start, min(x_min_token, config.vocab_size - 1))
        x_max_token = max(config.coord_vocab_start, min(x_max_token, config.vocab_size - 1))
        
        sequence.extend([class_token, x_min_token, x_max_token])
    
    # Add EOS token
    sequence.append(config.eos_token)
    
    return torch.tensor(sequence, dtype=torch.long)


def sequence_to_intervals(sequence: torch.Tensor, config: TimeSeriesPix2SeqConfig) -> List[Dict]:
    """
    Convert predicted sequence tokens back to intervals.
    
    Args:
        sequence: [L] - Token sequence
        config: Model configuration
    
    Returns:
        List of interval dicts
    """
    intervals = []
    
    i = 0
    while i < len(sequence):
        token = sequence[i].item()
        
        # Check for EOS
        if token == config.eos_token or token == config.pad_token:
            break
        
        # Check if we have enough tokens
        if i + 2 >= len(sequence):
            break
        
        # Decode class
        class_token = sequence[i].item()
        if class_token < config.class_vocab_start:
            i += 1
            continue
        class_id = class_token - config.class_vocab_start
        
        # Decode positions
        x_min_token = sequence[i + 1].item()
        x_max_token = sequence[i + 2].item()
        
        x_min = (x_min_token - config.coord_vocab_start) / config.quantization_bins
        x_max = (x_max_token - config.coord_vocab_start) / config.quantization_bins
        
        # Validate
        if 0.0 <= x_min < x_max <= 1.0 and class_id >= 0 and class_id < config.num_classes:
            intervals.append({
                'class_id': class_id,
                'x_min': x_min,
                'x_max': x_max
            })
        
        i += 3
    
    return intervals


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    config = TimeSeriesPix2SeqConfig(
        seq_len=336,
        input_features=1,
        window_size=16,
        num_classes=64,  # From hierarchical vocabulary
        max_intervals=50
    )
    
    print("="*80)
    print("TIME SERIES PIX2SEQ MODEL")
    print("="*80)
    print(f"Sequence length: {config.seq_len}")
    print(f"Window size: {config.window_size}")
    print(f"Number of windows: {config.num_windows}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Max sequence length: {config.max_seq_len}")
    
    # Create model
    model = TimeSeriesPix2Seq(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Example input
    batch_size = 4
    timeseries = torch.randn(batch_size, config.seq_len, config.input_features)
    
    # Example intervals
    intervals = [
        {'class_id': 20, 'x_min': 0.0, 'x_max': 0.4},    # UPTREND_SHORT
        {'class_id': 30, 'x_min': 0.35, 'x_max': 0.35},  # LOCAL_PEAK
        {'class_id': 23, 'x_min': 0.4, 'x_max': 0.7},    # DOWNTREND_SHORT
    ]
    
    # Convert to sequence
    seq = intervals_to_sequence(intervals, config)
    print(f"\nExample intervals: {len(intervals)}")
    print(f"Sequence length: {len(seq)}")
    print(f"Sequence tokens: {seq.tolist()}")
    
    # Batch the sequence
    seq_batch = seq.unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    print("\n" + "="*80)
    print("FORWARD PASS")
    print("="*80)
    logits = model(timeseries, seq_batch[:, :-1])  # Teacher forcing
    print(f"Input timeseries shape: {timeseries.shape}")
    print(f"Input sequence shape: {seq_batch[:, :-1].shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Inference
    print("\n" + "="*80)
    print("INFERENCE")
    print("="*80)
    model.eval()
    predictions = model.predict(timeseries, max_intervals=10, temperature=0.8, top_p=0.4)
    print(f"Predictions shape: {predictions.shape}")
    
    # Decode first prediction
    decoded = sequence_to_intervals(predictions[0], config)
    print(f"\nDecoded intervals: {len(decoded)}")
    for i, interval in enumerate(decoded):
        print(f"  {i+1}. Class {interval['class_id']}: "
              f"[{interval['x_min']:.3f}, {interval['x_max']:.3f}]")
    
    print("\n" + "="*80)
    print("✓ MODEL READY")
    print("="*80)