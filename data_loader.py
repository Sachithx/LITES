"""
HAR PIX2SEQ DATASET ADAPTER - CORRECT VERSION
==============================================

Workflow:
1. Input: Original HAR data [N, 9, 128] with labels [N]
2. Annotation: Split to [N×9, 128] univariate → Generate hierarchical annotations
3. Training: Reconstruct to [N, 9, 128] with grouped annotations

Each training sample:
    - timeseries: [9, 128] (original multi-channel)
    - annotations: List of 9 interval lists (one per channel)
    - har_label: Activity class (0-5)
    - target_sequence: Pix2Seq format including all 9 channels' intervals
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Pix2SeqConfig:
    """Configuration for Pix2Seq format."""
    
    num_activity_classes: int = 6   # HAR classes
    num_event_labels: int = 64      # Event vocabulary size
    quantization_bins: int = 1000   # Position discretization
    max_seq_len: int = 2048         # Max tokens (9 channels × ~100 events)
    
    # Token ranges
    pad_token: int = 0
    eos_token: int = 1
    har_class_start: int = 100      # 100-105
    event_label_start: int = 200    # 200-263
    coord_vocab_start: int = 1000   # 1000-1999
    
    @property
    def vocab_size(self) -> int:
        return 100 + self.num_activity_classes + self.num_event_labels + self.quantization_bins


# ============================================================================
# DATASET ADAPTER
# ============================================================================

class HARPix2SeqDataset(Dataset):
    """
    Adapts CompleteHierarchicalEventDataset for HAR Pix2Seq training.
    
    Input:
        - original_data: [N, 9, 128] original multi-channel sequences
        - hierarchical_dataset: [N×9, 128] univariate annotations
        - activity_labels: [N] HAR labels
    
    Output per sample:
        - timeseries: [9, 128]
        - intervals: List of 9 interval lists
        - har_label: int
        - target_sequence: [L] Pix2Seq tokens
    """
    
    def __init__(self,
                 original_data: torch.Tensor,           # [N, 9, 128]
                 hierarchical_dataset,                  # CompleteHierarchicalEventDataset [N×9]
                 activity_labels: torch.Tensor,         # [N]
                 n_channels: int = 9,
                 config: Optional[Pix2SeqConfig] = None,
                 verbose: bool = True):
        """
        Args:
            original_data: Original multi-channel data [N, C, L]
            hierarchical_dataset: Univariate annotations [N×C]
            activity_labels: HAR labels [N]
            n_channels: Number of channels
            config: Pix2Seq config
            verbose: Print info
        """
        super().__init__()
        
        self.original_data = original_data
        self.hierarchical_dataset = hierarchical_dataset
        self.activity_labels = activity_labels
        self.n_channels = n_channels
        self.config = config or Pix2SeqConfig()
        
        N, C, L = original_data.shape
        N_annotations = len(hierarchical_dataset)
        
        assert C == n_channels, f"Channel mismatch: {C} != {n_channels}"
        assert N_annotations == N * C, \
            f"Annotation count mismatch: {N_annotations} != {N} × {C}"
        assert len(activity_labels) == N, \
            f"Label count mismatch: {len(activity_labels)} != {N}"
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"CREATING HAR PIX2SEQ DATASET")
            print(f"{'='*80}")
            print(f"Original samples: {N}")
            print(f"Channels: {C}")
            print(f"Sequence length: {L}")
            print(f"Univariate annotations: {N_annotations}")
            print(f"HAR labels: {len(activity_labels)}")
        
        # Group annotations by sample and create target sequences
        if verbose:
            print(f"\nGrouping channel annotations and creating Pix2Seq sequences...")
        
        self.all_intervals = []  # [N] each element is list of C interval lists
        self.target_sequences = []
        
        for i in range(N):
            if verbose and i % 500 == 0:
                print(f"  Processing sample {i}/{N}...")
            
            # Get annotations for all C channels of sample i
            channel_intervals = []
            
            for c in range(C):
                ann_idx = i * C + c
                annotation = hierarchical_dataset[ann_idx]
                intervals = self._extract_intervals(annotation)
                channel_intervals.append(intervals)
            
            self.all_intervals.append(channel_intervals)
            
            # Create Pix2Seq target sequence
            target_seq = self._create_target_sequence(
                activity_labels[i].item(),
                channel_intervals,
                L
            )
            self.target_sequences.append(target_seq)
        
        if verbose:
            self._print_statistics(N, C)
    
    def _extract_intervals(self, annotation) -> List[Tuple[int, int, int]]:
        """Extract sorted intervals from annotation."""
        intervals = [(e.start, e.end, e.label) for e in annotation.all_events]
        intervals.sort(key=lambda x: x[0])
        return intervals
    
    def _create_target_sequence(self,
                                 har_label: int,
                                 channel_intervals: List[List[Tuple[int, int, int]]],
                                 seq_len: int) -> torch.Tensor:
        """
        Create Pix2Seq target sequence.
        
        Format: [har_class, event1, x_min, x_max, event2, ..., EOS]
        
        Args:
            har_label: HAR class (0-5)
            channel_intervals: List of C interval lists
            seq_len: Sequence length for normalization
        
        Returns:
            Token sequence [L]
        """
        sequence = []
        
        # 1. HAR class token
        har_token = self.config.har_class_start + har_label
        sequence.append(har_token)
        
        # 2. All intervals from all channels
        for channel_idx, intervals in enumerate(channel_intervals):
            for start, end, label_id in intervals:
                # Event label
                event_token = self.config.event_label_start + label_id
                
                # Coordinates (normalized)
                x_min = start / seq_len
                x_max = end / seq_len
                
                x_min_token = self.config.coord_vocab_start + int(x_min * self.config.quantization_bins)
                x_max_token = self.config.coord_vocab_start + int(x_max * self.config.quantization_bins)
                
                # Clamp to valid range
                x_min_token = min(x_min_token, self.config.coord_vocab_start + self.config.quantization_bins - 1)
                x_max_token = min(x_max_token, self.config.coord_vocab_start + self.config.quantization_bins - 1)
                
                sequence.extend([event_token, x_min_token, x_max_token])
        
        # 3. EOS token
        sequence.append(self.config.eos_token)
        
        return torch.tensor(sequence, dtype=torch.long)
    
    def _print_statistics(self, N: int, C: int):
        """Print dataset statistics."""
        total_intervals = sum(
            sum(len(ch_ints) for ch_ints in sample_ints)
            for sample_ints in self.all_intervals
        )
        
        avg_intervals_per_sample = total_intervals / N
        avg_intervals_per_channel = total_intervals / (N * C)
        avg_seq_len = sum(len(seq) for seq in self.target_sequences) / len(self.target_sequences)
        
        print(f"\n{'='*80}")
        print(f"✓ DATASET READY")
        print(f"{'='*80}")
        print(f"Samples: {N}")
        print(f"Total intervals (all channels): {total_intervals}")
        print(f"Avg intervals per sample: {avg_intervals_per_sample:.1f}")
        print(f"Avg intervals per channel: {avg_intervals_per_channel:.1f}")
        print(f"Avg target sequence length: {avg_seq_len:.1f} tokens")
        print(f"Vocabulary size: {self.config.vocab_size}")
    
    def __len__(self):
        return len(self.original_data)
    
    def __getitem__(self, idx):
        """
        Get one training sample.
        
        Returns:
            {
                'timeseries': [9, 128] - original multi-channel
                'intervals': List of 9 interval lists
                'har_label': int
                'target_sequence': [L] tokens
            }
        """
        return {
            'timeseries': self.original_data[idx],      # [C, L]
            'intervals': self.all_intervals[idx],        # List of C interval lists
            'har_label': self.activity_labels[idx],
            'target_sequence': self.target_sequences[idx]
        }


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def har_collate_fn(batch: List[Dict], pad_token: int = 0) -> Dict:
    """Collate with padding for variable-length sequences."""
    
    # Stack tensors
    timeseries = torch.stack([item['timeseries'] for item in batch])  # [B, C, L]
    har_labels = torch.stack([item['har_label'] for item in batch])   # [B]
    
    # Pad target sequences
    target_sequences = [item['target_sequence'] for item in batch]
    target_lengths = torch.tensor([len(seq) for seq in target_sequences])
    
    max_len = max(target_lengths).item()
    
    padded_sequences = []
    for seq in target_sequences:
        pad_len = max_len - len(seq)
        padded = F.pad(seq, (0, pad_len), value=pad_token)
        padded_sequences.append(padded)
    
    target_sequences = torch.stack(padded_sequences)
    
    # Keep intervals as list
    intervals = [item['intervals'] for item in batch]
    
    return {
        'timeseries': timeseries,           # [B, C, L]
        'intervals': intervals,              # List[List[List]] - B samples × C channels × intervals
        'har_label': har_labels,            # [B]
        'target_sequence': target_sequences, # [B, max_len]
        'target_length': target_lengths      # [B]
    }


# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_har_dataloader(
    original_data: torch.Tensor,        # [N, C, L]
    hierarchical_dataset,               # CompleteHierarchicalEventDataset [N×C]
    activity_labels: torch.Tensor,      # [N]
    batch_size: int = 32,
    n_channels: int = 9,
    config: Optional[Pix2SeqConfig] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = True
) -> DataLoader:
    """
    Create DataLoader for HAR Pix2Seq training.
    
    Args:
        original_data: Original multi-channel data [N, C, L]
        hierarchical_dataset: Univariate annotations [N×C]
        activity_labels: HAR labels [N]
        batch_size: Batch size
        n_channels: Number of channels
        config: Pix2Seq config
        shuffle: Shuffle data
        num_workers: DataLoader workers
        verbose: Print info
    
    Returns:
        DataLoader ready for training
    """
    if config is None:
        config = Pix2SeqConfig()
    
    # Create dataset
    dataset = HARPix2SeqDataset(
        original_data,
        hierarchical_dataset,
        activity_labels,
        n_channels=n_channels,
        config=config,
        verbose=verbose
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: har_collate_fn(b, config.pad_token),
        pin_memory=True
    )
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✓ DATALOADER READY")
        print(f"{'='*80}")
        print(f"Batches: {len(dataloader)}")
        print(f"Batch size: {batch_size}")
    
    return dataloader


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_sample(dataset: HARPix2SeqDataset, idx: int = 0):
    """Visualize one sample."""
    
    sample = dataset[idx]
    config = dataset.config
    
    activity_names = ['WALKING', 'UPSTAIRS', 'DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    
    print(f"\n{'='*80}")
    print(f"SAMPLE {idx}")
    print(f"{'='*80}")
    
    # Timeseries
    ts = sample['timeseries']
    print(f"\nTimeseries: {ts.shape}")
    print(f"Value range: [{ts.min():.3f}, {ts.max():.3f}]")
    
    # HAR label
    har_id = sample['har_label'].item()
    print(f"\nHAR Activity: {activity_names[har_id]}")
    
    # Intervals per channel
    intervals = sample['intervals']
    print(f"\nAnnotations per channel:")
    
    from hierarchical_events_complete import VOCAB
    
    for ch_idx, ch_intervals in enumerate(intervals):
        print(f"\n  Channel {ch_idx}: {len(ch_intervals)} intervals")
        
        if len(ch_intervals) > 0:
            print(f"  {'Start':>6} {'End':>6} {'Label':<25}")
            for start, end, label_id in ch_intervals[:5]:
                label_name = VOCAB.id_to_label(label_id)
                print(f"  {start:>6} {end:>6} {label_name:<25}")
            
            if len(ch_intervals) > 5:
                print(f"  ... and {len(ch_intervals) - 5} more")
    
    # Total intervals
    total_intervals = sum(len(ch_ints) for ch_ints in intervals)
    print(f"\nTotal intervals across all channels: {total_intervals}")
    
    # Target sequence
    seq = sample['target_sequence']
    print(f"\nTarget sequence length: {len(seq)} tokens")
    print(f"First 30 tokens: {seq[:30].tolist()}")
    
    # Decode
    print(f"\nDecoded (first 5 events):")
    tokens = seq.tolist()
    
    # HAR class
    print(f"  [0] HAR: {activity_names[tokens[0] - config.har_class_start]}")
    
    # Events
    i = 1
    count = 0
    while i < len(tokens) and tokens[i] != config.eos_token and count < 5:
        if i + 2 >= len(tokens):
            break
        
        event_id = tokens[i] - config.event_label_start
        x_min = (tokens[i+1] - config.coord_vocab_start) / config.quantization_bins
        x_max = (tokens[i+2] - config.coord_vocab_start) / config.quantization_bins
        
        event_name = VOCAB.id_to_label(event_id)
        count += 1
        print(f"  [{count}] {event_name}: [{x_min:.3f}, {x_max:.3f}]")
        
        i += 3
    
    if i < len(tokens) and tokens[i] != config.eos_token:
        print(f"  ... and more events")
    print(f"  [EOS]")

# ============================================================================