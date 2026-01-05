"""
HAR PIX2SEQ DATASET - SHUFFLED INTERVALS PER EPOCH
===================================================

Key features:
1. Timeseries: Fixed [9, 128]
2. Intervals: Shuffled order each epoch (per channel)
3. Target sequence: Flat tokens [BOS, start, end, label, ..., EOS]
4. NO HAR label in sequence (only annotations)
5. Each channel's intervals shuffled independently

Format: [BOS, start_1, end_1, label_1, start_2, end_2, label_2, ..., EOS]
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ShuffledPix2SeqConfig:
    """Configuration for shuffled interval Pix2Seq format."""
    
    seq_len: int = 128              # Sequence length
    n_channels: int = 9             # Number of channels
    num_event_labels: int = 64      # Event vocabulary
    
    # Token IDs
    pad_token: int = 0
    bos_token: int = 1              # BOS instead of HAR class
    eos_token: int = 2
    
    # Token ranges (simplified - no HAR class tokens)
    event_label_start: int = 100    # Event labels: 100-163
    position_start: int = 200       # Positions: 200-327 (for 0-127)
    
    max_seq_len: int = 2048         # Max tokens
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary: special(3) + events(64) + positions(128) = 195."""
        return 3 + self.num_event_labels + self.seq_len
    
    def event_to_token(self, event_id: int) -> int:
        """Convert event ID to token."""
        return self.event_label_start + event_id
    
    def position_to_token(self, pos: int) -> int:
        """Convert position (0-127) to token."""
        return self.position_start + pos


# ============================================================================
# SHUFFLED INTERVAL DATASET
# ============================================================================

class ShuffledIntervalDataset(Dataset):
    """
    HAR dataset with shuffled interval order per epoch.
    
    Each epoch shuffles intervals independently for each channel.
    Target sequence: [BOS, start, end, label, start, end, label, ..., EOS]
    """
    
    def __init__(self,
                 original_data: torch.Tensor,           # [N, C, L]
                 hierarchical_dataset,                  # [N×C] univariate annotations
                 n_channels: int = 9,
                 config: Optional[ShuffledPix2SeqConfig] = None,
                 shuffle_on_init: bool = True,
                 verbose: bool = True):
        """
        Args:
            original_data: Original multi-channel [N, C, L]
            hierarchical_dataset: Univariate annotations [N×C]
            n_channels: Number of channels
            config: Configuration
            shuffle_on_init: Shuffle intervals on initialization
            verbose: Print info
        """
        super().__init__()
        
        self.original_data = original_data
        self.hierarchical_dataset = hierarchical_dataset
        self.n_channels = n_channels
        self.config = config or ShuffledPix2SeqConfig()
        
        N, C, L = original_data.shape
        N_annotations = len(hierarchical_dataset)
        
        assert C == n_channels
        assert N_annotations == N * C
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"CREATING SHUFFLED INTERVAL PIX2SEQ DATASET")
            print(f"{'='*80}")
            print(f"Samples: {N}")
            print(f"Channels: {C}")
            print(f"Sequence length: {L}")
            print(f"Annotations: {N_annotations}")
            print(f"Vocab size: {self.config.vocab_size}")
        
        # Extract all intervals (will be shuffled per epoch)
        if verbose:
            print(f"\nExtracting intervals from annotations...")
        
        self.all_intervals = []  # [N] each is list of C channel intervals
        
        for i in range(N):
            if verbose and i % 500 == 0:
                print(f"  Processing sample {i}/{N}...")
            
            channel_intervals = []
            for c in range(C):
                ann_idx = i * C + c
                annotation = hierarchical_dataset[ann_idx]
                intervals = self._extract_intervals(annotation)
                channel_intervals.append(intervals)
            
            self.all_intervals.append(channel_intervals)
        
        # Compute statistics
        if verbose:
            self._print_statistics(N, C)
        
        # Shuffle on init if requested
        if shuffle_on_init:
            if verbose:
                print(f"\nNote: Intervals will be shuffled on-the-fly per sample access")
                print(f"      (no need for epoch-level shuffling)")
    
    def _extract_intervals(self, annotation) -> List[Tuple[int, int, int]]:
        """Extract intervals from annotation."""
        intervals = [(e.start, e.end, e.label) for e in annotation.all_events]
        return intervals  # Don't sort - will be shuffled per access
    
    def _print_statistics(self, N: int, C: int):
        """Print dataset statistics."""
        total_intervals = sum(
            sum(len(ch_ints) for ch_ints in sample_ints)
            for sample_ints in self.all_intervals
        )
        
        avg_per_sample = total_intervals / N
        avg_per_channel = total_intervals / (N * C)
        
        # Estimate sequence length (3 tokens per interval + BOS + EOS)
        avg_seq_len = avg_per_sample * 3 + 2
        
        print(f"\n{'='*80}")
        print(f"✓ DATASET READY")
        print(f"{'='*80}")
        print(f"Samples: {N}")
        print(f"Total intervals: {total_intervals}")
        print(f"Avg intervals per sample: {avg_per_sample:.1f}")
        print(f"Avg intervals per channel: {avg_per_channel:.1f}")
        print(f"Estimated avg sequence length: {avg_seq_len:.1f} tokens")
    
    def shuffle_all(self):
        """Shuffle intervals for all samples and channels."""
        for sample_intervals in self.all_intervals:
            for channel_intervals in sample_intervals:
                random.shuffle(channel_intervals)
    
    def _create_target_sequence(self, 
                                 channel_intervals: List[List[Tuple[int, int, int]]]) -> torch.Tensor:
        """
        Create flat token sequence with shuffled intervals.
        
        Format: [BOS, start_1, end_1, label_1, start_2, end_2, label_2, ..., EOS]
        
        Args:
            channel_intervals: List of C interval lists (already shuffled)
        
        Returns:
            Token sequence [L]
        """
        sequence = [self.config.bos_token]  # Start with BOS
        
        # Add all intervals from all channels
        for channel_idx, intervals in enumerate(channel_intervals):
            for start, end, label_id in intervals:
                # Convert to tokens
                start_token = self.config.position_to_token(start)
                end_token = self.config.position_to_token(end)
                label_token = self.config.event_to_token(label_id)
                
                # Add triplet
                sequence.extend([start_token, end_token, label_token])
        
        # End with EOS
        sequence.append(self.config.eos_token)
        
        return torch.tensor(sequence, dtype=torch.long)
    
    def __len__(self):
        return len(self.original_data)
    
    def __getitem__(self, idx):
        """
        Get one sample with on-the-fly shuffling.
        
        IMPORTANT: Intervals are shuffled EVERY TIME this sample is accessed!
        This means:
        - Different shuffle in each epoch
        - Different shuffle in each batch
        - Same timeseries, different interval order
        
        Returns:
            {
                'timeseries': [C, L],
                'intervals': List of C interval lists (freshly shuffled),
                'target_sequence': [L] flat tokens
            }
        """
        # Get original intervals and shuffle them ON-THE-FLY
        shuffled_intervals = []
        for channel_intervals in self.all_intervals[idx]:
            # Create a copy and shuffle it
            shuffled = channel_intervals.copy()
            random.shuffle(shuffled)
            shuffled_intervals.append(shuffled)
        
        # Create target sequence with this shuffle
        target_seq = self._create_target_sequence(shuffled_intervals)
        
        return {
            'timeseries': self.original_data[idx],
            'intervals': shuffled_intervals,
            'target_sequence': target_seq
        }


# ============================================================================
# EPOCH SHUFFLING SAMPLER/DATALOADER
# ============================================================================

class ShuffledIntervalDataLoader:
    """
    DataLoader wrapper for on-the-fly interval shuffling.
    
    Intervals are shuffled EVERY TIME a sample is accessed,
    not once per epoch. This means:
    - Same sample in different batches → different shuffle
    - Same sample in different epochs → different shuffle
    
    Usage:
        loader = ShuffledIntervalDataLoader(dataset, batch_size=32)
        
        for epoch in range(num_epochs):
            for batch in loader:
                # Each sample has freshly shuffled intervals
                train_step(batch)
    """
    
    def __init__(self,
                 dataset: ShuffledIntervalDataset,
                 batch_size: int = 32,
                 shuffle_samples: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 verbose: bool = False):
        """
        Args:
            dataset: ShuffledIntervalDataset
            batch_size: Batch size
            shuffle_samples: Shuffle sample order (not interval order)
            num_workers: DataLoader workers
            pin_memory: Pin memory
            verbose: Print info
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_samples = shuffle_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose
        
        self._create_dataloader()
    
    def _create_dataloader(self):
        """Create PyTorch DataLoader."""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_samples,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Collate with padding."""
        # Stack timeseries
        timeseries = torch.stack([item['timeseries'] for item in batch])
        
        # Pad target sequences
        target_sequences = [item['target_sequence'] for item in batch]
        target_lengths = torch.tensor([len(seq) for seq in target_sequences])
        
        max_len = max(target_lengths).item()
        
        padded_sequences = []
        for seq in target_sequences:
            pad_len = max_len - len(seq)
            padded = F.pad(seq, (0, pad_len), value=self.dataset.config.pad_token)
            padded_sequences.append(padded)
        
        target_sequences = torch.stack(padded_sequences)
        
        # Keep intervals as list
        intervals = [item['intervals'] for item in batch]
        
        return {
            'timeseries': timeseries,
            'intervals': intervals,
            'target_sequence': target_sequences,
            'target_length': target_lengths
        }
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_shuffled_dataloader(
    original_data: torch.Tensor,
    hierarchical_dataset,
    batch_size: int = 32,
    n_channels: int = 9,
    config: Optional[ShuffledPix2SeqConfig] = None,
    shuffle_samples: bool = True,
    num_workers: int = 4,
    verbose: bool = True
) -> ShuffledIntervalDataLoader:
    """
    Create dataloader with per-epoch interval shuffling.
    
    Args:
        original_data: [N, C, L]
        hierarchical_dataset: [N×C]
        batch_size: Batch size
        n_channels: Number of channels
        config: Configuration
        shuffle_samples: Shuffle sample order
        num_workers: Workers
        verbose: Print info
    
    Returns:
        ShuffledIntervalDataLoader
    """
    if config is None:
        config = ShuffledPix2SeqConfig(
            seq_len=original_data.shape[2],
            n_channels=n_channels
        )
    
    # Create dataset
    dataset = ShuffledIntervalDataset(
        original_data,
        hierarchical_dataset,
        n_channels=n_channels,
        config=config,
        shuffle_on_init=True,
        verbose=verbose
    )
    
    # Create dataloader with epoch shuffling
    dataloader = ShuffledIntervalDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle_samples=shuffle_samples,
        num_workers=num_workers,
        verbose=verbose
    )
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✓ SHUFFLED DATALOADER READY")
        print(f"{'='*80}")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Shuffling: On-the-fly per sample access")
        print(f"           (Different shuffle each time sample is loaded)")
    
    return dataloader
