"""
USAGE GUIDE: Hierarchical Time Series Event Labeling System
============================================================

Quick Start Guide and Common Use Cases
"""

import torch
from hierarchical_event_labeling import (
    HierarchicalEventDataset,
    TextCorpusGenerator,
    EventScale,
    VOCAB
)


# ============================================================================
# QUICK START
# ============================================================================

def quick_start_example():
    """Minimal example to get started"""
    
    # 1. Prepare your data as [B, L] tensor
    B, L = 100, 336  # 100 sequences, each 336 timesteps
    x = torch.randn(B, L)  # Replace with your real data
    
    # 2. Create dataset (this does all the processing)
    dataset = HierarchicalEventDataset(x, verbose=True)
    
    # 3. Access annotations
    annotation = dataset[0]  # Get first sequence annotation
    
    # 4. Generate training text
    text = annotation.to_text(format='depth_marked')
    print(text)
    
    return dataset


# ============================================================================
# LOADING REAL DATA
# ============================================================================

def load_from_numpy():
    """Load from numpy arrays"""
    import numpy as np
    
    # Load your numpy data
    data = np.load('your_data.npy')  # Shape: [num_samples, sequence_length]
    
    # Convert to tensor
    x = torch.from_numpy(data).float()
    
    # Create dataset
    dataset = HierarchicalEventDataset(x)
    
    return dataset


def load_from_csv():
    """Load from CSV files"""
    import pandas as pd
    
    # Load CSV
    df = pd.read_csv('your_data.csv')
    
    # Assuming each row is a sequence
    data = df.values  # Shape: [num_sequences, sequence_length]
    x = torch.from_numpy(data).float()
    
    dataset = HierarchicalEventDataset(x)
    
    return dataset


def load_eeg_example():
    """Example for EEG data"""
    
    # Assuming you have EEG data: [num_trials, num_channels, time_points]
    eeg_data = torch.randn(100, 64, 1000)  # Replace with real data
    
    # Process each channel separately
    datasets = []
    for channel in range(64):
        channel_data = eeg_data[:, channel, :]  # [num_trials, time_points]
        dataset = HierarchicalEventDataset(channel_data, verbose=False)
        datasets.append(dataset)
    
    return datasets


# ============================================================================
# EXPLORING ANNOTATIONS
# ============================================================================

def explore_annotation(dataset):
    """Explore annotation structure"""
    
    ann = dataset[0]
    
    print("="*80)
    print("ANNOTATION EXPLORATION")
    print("="*80)
    
    # 1. View hierarchical structure
    print("\n1. HIERARCHICAL TREE:")
    ann.print_hierarchy(max_depth=3)
    
    # 2. Get events at specific scale
    print("\n2. EVENTS BY SCALE:")
    for scale in EventScale:
        events = ann.get_events_at_scale(scale)
        print(f"{scale.name}: {len(events)} events")
    
    # 3. Get events in time range
    print("\n3. EVENTS IN TIME RANGE [100-200]:")
    events = ann.get_events_in_range(100, 200)
    for e in events:
        print(f"  [{e.start:03d}-{e.end:03d}] {e.label_name} ({e.scale.name})")
    
    # 4. Access event metadata
    print("\n4. EVENT METADATA:")
    for event in ann.all_events[:5]:
        print(f"{event.label_name}: {event.metadata}")
    
    # 5. Step-wise labels
    print(f"\n5. STEP-WISE LABELS (first 20):")
    print(f"IDs: {ann.step_labels[:20].tolist()}")
    print(f"Names: {[VOCAB.id_to_label(int(l)) for l in ann.step_labels[:20]]}")


# ============================================================================
# TEXT GENERATION FOR TRAINING
# ============================================================================

def generate_training_corpus(dataset, output_file='training_corpus.txt'):
    """Generate complete training corpus"""
    
    # Generate text for all sequences
    text_gen = TextCorpusGenerator()
    
    # Try different formats
    formats = ['depth_marked', 'flat', 'narrative']
    
    for fmt in formats:
        print(f"\nGenerating {fmt} format...")
        corpus = text_gen.generate_corpus(dataset, format=fmt)
        
        # Save to file
        filename = f'{fmt}_{output_file}'
        with open(filename, 'w') as f:
            for i, text in enumerate(corpus):
                f.write(f"<sequence_{i}>\n{text}\n</sequence_{i}>\n\n")
        
        # Print statistics
        stats = text_gen.estimate_tokens(corpus)
        print(f"Saved to {filename}")
        print(f"  Documents: {stats['num_documents']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")


def create_autoregressive_pairs(dataset):
    """Create input-output pairs for autoregressive LM training"""
    
    pairs = []
    
    for ann in dataset.annotations:
        # Get hierarchical text
        full_text = ann.to_text(format='depth_marked')
        tokens = full_text.split()
        
        # Create prefix-completion pairs
        for i in range(1, len(tokens)):
            input_text = " ".join(tokens[:i])
            target_token = tokens[i]
            pairs.append((input_text, target_token))
    
    return pairs


# ============================================================================
# FILTERING AND ANALYSIS
# ============================================================================

def filter_by_event_type(dataset, event_type='trend'):
    """Filter sequences by event type"""
    
    filtered = []
    for i, ann in enumerate(dataset.annotations):
        # Check if this annotation contains the event type
        has_event = any(e.event_type == event_type for e in ann.all_events)
        if has_event:
            filtered.append(i)
    
    print(f"Found {len(filtered)} sequences with {event_type} events")
    return filtered


def analyze_event_statistics(dataset):
    """Compute detailed event statistics"""
    
    stats = {
        'total_sequences': len(dataset),
        'events_by_type': {},
        'events_by_scale': {},
        'events_by_label': {},
    }
    
    # Count events
    for ann in dataset.annotations:
        for event in ann.all_events:
            # By type
            stats['events_by_type'][event.event_type] = \
                stats['events_by_type'].get(event.event_type, 0) + 1
            
            # By scale
            stats['events_by_scale'][event.scale.name] = \
                stats['events_by_scale'].get(event.scale.name, 0) + 1
            
            # By label
            stats['events_by_label'][event.label_name] = \
                stats['events_by_label'].get(event.label_name, 0) + 1
    
    # Print statistics
    print("\n" + "="*80)
    print("DETAILED EVENT STATISTICS")
    print("="*80)
    
    print("\nBy Event Type:")
    for event_type, count in sorted(stats['events_by_type'].items()):
        avg = count / stats['total_sequences']
        print(f"  {event_type:.<20} {count:>6} ({avg:.2f} per sequence)")
    
    print("\nBy Scale:")
    for scale, count in sorted(stats['events_by_scale'].items()):
        avg = count / stats['total_sequences']
        print(f"  {scale:.<20} {count:>6} ({avg:.2f} per sequence)")
    
    print("\nTop 10 Labels:")
    top_labels = sorted(stats['events_by_label'].items(), 
                       key=lambda x: x[1], reverse=True)[:10]
    for label, count in top_labels:
        avg = count / stats['total_sequences']
        print(f"  {label:.<30} {count:>6} ({avg:.2f} per sequence)")
    
    return stats


# ============================================================================
# CUSTOM TEXT FORMATS
# ============================================================================

def create_custom_format(ann):
    """Create your own text format"""
    
    parts = []
    
    # Add sequence metadata
    parts.append(f"SEQ_LEN:{len(ann.sequence)}")
    
    # Add global regime
    global_events = ann.get_events_at_scale(EventScale.GLOBAL)
    if global_events:
        parts.append(f"REGIME:{global_events[0].label_name}")
    
    # Add macro trends
    macro_events = ann.get_events_at_scale(EventScale.MACRO)
    parts.append(f"TRENDS:{len(macro_events)}")
    for event in macro_events:
        parts.append(f"T[{event.start}-{event.end}]:{event.label_name}")
    
    # Add peaks
    peaks = [e for e in ann.all_events if e.event_type == 'peak']
    parts.append(f"PEAKS:{len(peaks)}")
    for peak in peaks:
        parts.append(f"P[{peak.start}]:{peak.label_name}")
    
    return " | ".join(parts)


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_large_dataset_in_batches(data_generator, batch_size=1000):
    """Process very large datasets in batches"""
    
    all_annotations = []
    
    for batch_idx, batch_data in enumerate(data_generator):
        print(f"\nProcessing batch {batch_idx}...")
        
        # Create dataset for this batch
        dataset = HierarchicalEventDataset(batch_data, verbose=False)
        
        # Collect annotations
        all_annotations.extend(dataset.annotations)
        
        # Optionally save intermediate results
        torch.save(dataset.annotations, f'annotations_batch_{batch_idx}.pt')
    
    print(f"\nTotal annotations: {len(all_annotations)}")
    return all_annotations


# ============================================================================
# INTEGRATION WITH TRAINING PIPELINES
# ============================================================================

def create_pytorch_dataloader(dataset, batch_size=32):
    """Create PyTorch DataLoader for training"""
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function for hierarchical annotations"""
        sequences = torch.stack([ann.sequence for ann in batch])
        step_labels = torch.stack([ann.step_labels for ann in batch])
        
        # Generate text representations
        texts = [ann.to_text(format='depth_marked') for ann in batch]
        
        return {
            'sequences': sequences,
            'step_labels': step_labels,
            'texts': texts,
            'annotations': batch  # Keep full annotations if needed
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def prepare_for_huggingface(dataset, tokenizer):
    """Prepare data for HuggingFace transformers"""
    
    # Generate text corpus
    text_gen = TextCorpusGenerator()
    texts = text_gen.generate_corpus(dataset, format='depth_marked')
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    return tokenized


# ============================================================================
# EXAMPLE WORKFLOWS
# ============================================================================

def complete_workflow_example():
    """Complete end-to-end workflow"""
    
    print("\n" + "="*80)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("="*80)
    
    # 1. Generate/Load data
    print("\n[1/6] Loading data...")
    B, L = 100, 336
    x = torch.randn(B, L)  # Replace with real data
    
    # 2. Create dataset
    print("\n[2/6] Creating hierarchical dataset...")
    dataset = HierarchicalEventDataset(x, verbose=False)
    print(f"  ✓ Processed {len(dataset)} sequences")
    
    # 3. Explore one example
    print("\n[3/6] Exploring example annotation...")
    explore_annotation(dataset)
    
    # 4. Analyze statistics
    print("\n[4/6] Computing statistics...")
    stats = analyze_event_statistics(dataset)
    
    # 5. Generate training corpus
    print("\n[5/6] Generating training corpus...")
    generate_training_corpus(dataset, 'output_corpus.txt')
    
    # 6. Create DataLoader
    print("\n[6/6] Creating DataLoader...")
    dataloader = create_pytorch_dataloader(dataset, batch_size=16)
    print(f"  ✓ DataLoader ready with {len(dataloader)} batches")
    
    print("\n" + "="*80)
    print("✓ WORKFLOW COMPLETE")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run the complete workflow
    complete_workflow_example()