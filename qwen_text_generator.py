"""
Qwen-Optimized Text Generation for Hierarchical Time Series Events
===================================================================

Extends the hierarchical event labeling system with optimized text generation
for Qwen tokenizer and transformer training.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# VOCABULARY MAPPING TO NATURAL LANGUAGE
# ============================================================================

class VocabularyMapper:
    """Maps event vocabulary to natural language descriptions"""
    
    # Complete mapping of all 64 event types
    VOCAB_TO_NATURAL = {
        # Special tokens
        'PAD': '',
        'MASK': '[masked]',
        'FLAT': 'stable',
        
        # Step movements
        'UP_SMALL': 'small rise',
        'UP_MEDIUM': 'moderate rise',
        'UP_LARGE': 'large rise',
        'DOWN_SMALL': 'small decline',
        'DOWN_MEDIUM': 'moderate decline',
        'DOWN_LARGE': 'large decline',
        'SPIKE_UP': 'sharp upward spike',
        'SPIKE_DOWN': 'sharp downward spike',
        
        # Trend segments
        'UPTREND_SHORT': 'short upward trend',
        'UPTREND_MEDIUM': 'medium upward trend',
        'UPTREND_LONG': 'long upward trend',
        'DOWNTREND_SHORT': 'short downward trend',
        'DOWNTREND_MEDIUM': 'medium downward trend',
        'DOWNTREND_LONG': 'long downward trend',
        'FLAT_SEGMENT': 'flat stable segment',
        
        # Peaks and troughs
        'LOCAL_PEAK': 'local peak',
        'SHARP_PEAK': 'sharp prominent peak',
        'LOCAL_TROUGH': 'local trough',
        'SHARP_TROUGH': 'sharp deep trough',
        
        # Volatility
        'LOW_VOLATILITY': 'low volatility period',
        'NORMAL_VOLATILITY': 'normal volatility',
        'HIGH_VOLATILITY': 'high volatility period',
        'VOLATILITY_SPIKE': 'sudden volatility spike',
        
        # Change points
        'MEAN_SHIFT_UP': 'upward level shift',
        'MEAN_SHIFT_DOWN': 'downward level shift',
        
        # Global regimes
        'BULLISH_REGIME': 'bullish upward regime',
        'BEARISH_REGIME': 'bearish downward regime',
        'SIDEWAYS_REGIME': 'sideways consolidation regime',
        'VOLATILE_REGIME': 'highly volatile regime',
    }
    
    # Alternative compact codes (for Strategy 5)
    VOCAB_TO_COMPACT = {
        'FLAT': 'FL',
        'UP_SMALL': 'U1', 'UP_MEDIUM': 'U2', 'UP_LARGE': 'U3',
        'DOWN_SMALL': 'D1', 'DOWN_MEDIUM': 'D2', 'DOWN_LARGE': 'D3',
        'SPIKE_UP': 'SU', 'SPIKE_DOWN': 'SD',
        'UPTREND_SHORT': 'UT1', 'UPTREND_MEDIUM': 'UT2', 'UPTREND_LONG': 'UT3',
        'DOWNTREND_SHORT': 'DT1', 'DOWNTREND_MEDIUM': 'DT2', 'DOWNTREND_LONG': 'DT3',
        'FLAT_SEGMENT': 'FLT',
        'LOCAL_PEAK': 'PK', 'SHARP_PEAK': 'PKS',
        'LOCAL_TROUGH': 'TR', 'SHARP_TROUGH': 'TRS',
        'LOW_VOLATILITY': 'VL', 'NORMAL_VOLATILITY': 'VN',
        'HIGH_VOLATILITY': 'VH', 'VOLATILITY_SPIKE': 'VS',
        'BULLISH_REGIME': 'RB', 'BEARISH_REGIME': 'RR',
        'SIDEWAYS_REGIME': 'RS', 'VOLATILE_REGIME': 'RV',
    }
    
    @classmethod
    def to_natural(cls, label: str) -> str:
        """Convert label to natural language"""
        return cls.VOCAB_TO_NATURAL.get(label, label.lower().replace('_', ' '))
    
    @classmethod
    def to_compact(cls, label: str) -> str:
        """Convert label to compact code"""
        return cls.VOCAB_TO_COMPACT.get(label, label)


# ============================================================================
# QWEN-OPTIMIZED TEXT GENERATORS
# ============================================================================

class QwenTextGenerator:
    """Generate Qwen-optimized text from hierarchical annotations"""
    
    def __init__(self, strategy: str = 'natural_hierarchical'):
        """
        Args:
            strategy: 'natural_hierarchical', 'natural_flat', 'compact', 'hybrid'
        """
        self.strategy = strategy
        self.mapper = VocabularyMapper()
    
    def generate(self, annotation) -> str:
        """
        Generate text from HierarchicalAnnotation
        
        Args:
            annotation: HierarchicalAnnotation object
        
        Returns:
            Text string optimized for Qwen tokenizer
        """
        if self.strategy == 'natural_hierarchical':
            return self._natural_hierarchical(annotation)
        elif self.strategy == 'natural_flat':
            return self._natural_flat(annotation)
        elif self.strategy == 'compact':
            return self._compact(annotation)
        elif self.strategy == 'hybrid':
            return self._hybrid(annotation)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _natural_hierarchical(self, ann) -> str:
        """
        Natural language with explicit hierarchy
        
        Example:
        "[0-335] The signal exhibits sideways consolidation regime. 
         [0-120] There is a long upward trend. 
         [50] At position 50, a local peak occurs."
        """
        parts = []
        
        # Global context
        global_events = [e for e in ann.all_events if e.scale.name == 'GLOBAL']
        if global_events:
            e = global_events[0]
            desc = self.mapper.to_natural(e.label_name)
            parts.append(f"[{e.start}-{e.end}] The signal exhibits {desc}.")
        
        # Macro-level events
        macro_events = [e for e in ann.all_events if e.scale.name == 'MACRO']
        for e in macro_events:
            desc = self.mapper.to_natural(e.label_name)
            parts.append(f"[{e.start}-{e.end}] There is a {desc}.")
            
            # Mention important children
            if e.children:
                peaks = [c for c in e.children if 'peak' in c.label_name.lower()]
                troughs = [c for c in e.children if 'trough' in c.label_name.lower()]
                
                if peaks:
                    peak_desc = self.mapper.to_natural(peaks[0].label_name)
                    parts.append(f"[{peaks[0].start}] At position {peaks[0].start}, a {peak_desc} occurs.")
                
                if troughs:
                    trough_desc = self.mapper.to_natural(troughs[0].label_name)
                    parts.append(f"[{troughs[0].start}] At position {troughs[0].start}, a {trough_desc} occurs.")
        
        # Meso-level volatility (if significant)
        meso_vol = [e for e in ann.all_events 
                   if e.scale.name == 'MESO' and 'volatility' in e.label_name.lower()]
        for e in meso_vol:
            desc = self.mapper.to_natural(e.label_name)
            parts.append(f"[{e.start}-{e.end}] During this period, {desc}.")
        
        return " ".join(parts)
    
    def _natural_flat(self, ann) -> str:
        """
        Natural language, flat chronological order
        
        Example:
        "[0-335] sideways consolidation regime, 
         [0-120] long upward trend, 
         [50] local peak, 
         [120-200] flat stable segment"
        """
        parts = []
        
        # Sort by start position
        sorted_events = sorted(ann.all_events, key=lambda e: (e.start, -e.scale))
        
        for e in sorted_events:
            desc = self.mapper.to_natural(e.label_name)
            if e.start == e.end:
                parts.append(f"[{e.start}] {desc}")
            else:
                parts.append(f"[{e.start}-{e.end}] {desc}")
        
        return ", ".join(parts)
    
    def _compact(self, ann) -> str:
        """
        Compact depth-marked format
        
        Example:
        "RS:0-335 >UT3:0-120 >>PK:50 >>TR:75 >DT3:201-335"
        """
        def format_event(e, depth=0):
            prefix = '>' * depth
            code = self.mapper.to_compact(e.label_name)
            span = f":{e.start}-{e.end}" if e.start != e.end else f":{e.start}"
            return f"{prefix}{code}{span}"
        
        def traverse(node, depth=0):
            parts = [format_event(node, depth)]
            for child in node.children:
                parts.extend(traverse(child, depth + 1))
            return parts
        
        all_parts = []
        for root in ann.event_roots:
            all_parts.extend(traverse(root))
        
        return " ".join(all_parts)
    
    def _hybrid(self, ann) -> str:
        """
        Hybrid: Natural language for major events, compact for details
        
        Example:
        "[0-335] Sideways regime. Major segments: UT3:0-120 (PK:50), DT3:201-335 (TR:250)"
        """
        parts = []
        
        # Global in natural language
        global_events = [e for e in ann.all_events if e.scale.name == 'GLOBAL']
        if global_events:
            e = global_events[0]
            desc = self.mapper.to_natural(e.label_name)
            parts.append(f"[{e.start}-{e.end}] {desc.capitalize()}.")
        
        # Macro in hybrid format
        macro_events = [e for e in ann.all_events if e.scale.name == 'MACRO']
        if macro_events:
            macro_parts = []
            for e in macro_events:
                code = self.mapper.to_compact(e.label_name)
                span = f"{e.start}-{e.end}"
                
                # Add children
                child_codes = [self.mapper.to_compact(c.label_name) + f":{c.start}" 
                              for c in e.children[:3]]  # Limit to 3
                
                if child_codes:
                    macro_parts.append(f"{code}:{span} ({', '.join(child_codes)})")
                else:
                    macro_parts.append(f"{code}:{span}")
            
            parts.append(f"Major segments: {', '.join(macro_parts)}.")
        
        return " ".join(parts)


# ============================================================================
# BATCH CORPUS GENERATION
# ============================================================================

class QwenCorpusGenerator:
    """Generate complete training corpus for Qwen"""
    
    def __init__(self, strategy: str = 'natural_hierarchical'):
        self.generator = QwenTextGenerator(strategy=strategy)
    
    def generate_corpus(self, dataset, add_metadata: bool = True) -> List[str]:
        """
        Generate corpus from HierarchicalEventDataset
        
        Args:
            dataset: HierarchicalEventDataset instance
            add_metadata: Include sequence metadata (length, num events)
        
        Returns:
            List of text strings
        """
        corpus = []
        
        for ann in dataset.annotations:
            text = self.generator.generate(ann)
            
            if add_metadata:
                # Prepend metadata
                metadata = f"<sequence length={len(ann.sequence)} events={len(ann.all_events)}> "
                text = metadata + text
            
            corpus.append(text)
        
        return corpus
    
    def save_corpus(self, corpus: List[str], filepath: str, 
                   format: str = 'plain'):
        """
        Save corpus to file
        
        Args:
            corpus: List of text strings
            filepath: Output file path
            format: 'plain', 'jsonl', 'parquet'
        """
        if format == 'plain':
            with open(filepath, 'w', encoding='utf-8') as f:
                for text in corpus:
                    f.write(text + '\n')
        
        elif format == 'jsonl':
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                for i, text in enumerate(corpus):
                    obj = {'id': i, 'text': text}
                    f.write(json.dumps(obj) + '\n')
        
        elif format == 'parquet':
            import pandas as pd
            df = pd.DataFrame({'text': corpus})
            df.to_parquet(filepath, index=False)
    
    def estimate_tokens(self, corpus: List[str], tokenizer=None) -> Dict:
        """
        Estimate token counts for corpus
        
        Args:
            corpus: List of text strings
            tokenizer: Optional HuggingFace tokenizer
        
        Returns:
            Statistics dictionary
        """
        if tokenizer is None:
            # Rough estimation (words ≈ 1.3 tokens for English)
            total_words = sum(len(text.split()) for text in corpus)
            estimated_tokens = int(total_words * 1.3)
            
            stats = {
                'num_documents': len(corpus),
                'total_words': total_words,
                'estimated_tokens': estimated_tokens,
                'avg_words_per_doc': total_words / len(corpus),
                'avg_tokens_per_doc': estimated_tokens / len(corpus),
                'estimation_method': 'word_count * 1.3'
            }
        else:
            # Actual tokenization
            total_tokens = 0
            for text in corpus:
                tokens = tokenizer.tokenize(text)
                total_tokens += len(tokens)
            
            stats = {
                'num_documents': len(corpus),
                'total_tokens': total_tokens,
                'avg_tokens_per_doc': total_tokens / len(corpus),
                'tokenizer': tokenizer.name_or_path,
                'estimation_method': 'actual_tokenization'
            }
        
        return stats


# ============================================================================
# COMPLETE EXAMPLE WORKFLOW
# ============================================================================

def example_qwen_workflow():
    """
    Complete example: From raw data to Qwen training corpus
    """
    
    print("="*80)
    print("QWEN-OPTIMIZED CORPUS GENERATION WORKFLOW")
    print("="*80)
    
    # Step 1: Import and create dataset
    print("\n[1/5] Creating hierarchical event dataset...")
    from hierarchical_event_labeling import HierarchicalEventDataset
    import torch
    
    # Simulate data (replace with your real data)
    x = torch.randn(100, 336)
    dataset = HierarchicalEventDataset(x, verbose=False)
    print(f"  ✓ Processed {len(dataset)} sequences")
    
    # Step 2: Generate corpus with different strategies
    print("\n[2/5] Generating corpus with different strategies...")
    
    strategies = ['natural_hierarchical', 'natural_flat', 'compact', 'hybrid']
    
    for strategy in strategies:
        generator = QwenCorpusGenerator(strategy=strategy)
        corpus = generator.generate_corpus(dataset, add_metadata=True)
        
        # Show example
        print(f"\n  Strategy: {strategy}")
        print(f"  Example: {corpus[0][:150]}...")
        
        # Estimate tokens
        stats = generator.estimate_tokens(corpus)
        print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
        print(f"  Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")
    
    # Step 3: Recommended strategy
    print("\n[3/5] Using recommended strategy (natural_hierarchical)...")
    generator = QwenCorpusGenerator(strategy='natural_hierarchical')
    corpus = generator.generate_corpus(dataset, add_metadata=True)
    
    # Step 4: Save corpus
    print("\n[4/5] Saving corpus...")
    generator.save_corpus(corpus, 'qwen_training_corpus.txt', format='plain')
    generator.save_corpus(corpus, 'qwen_training_corpus.jsonl', format='jsonl')
    print("  ✓ Saved to qwen_training_corpus.txt")
    print("  ✓ Saved to qwen_training_corpus.jsonl")
    
    # Step 5: Summary
    print("\n[5/5] Summary...")
    stats = generator.estimate_tokens(corpus)
    print(f"  Total documents: {stats['num_documents']}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
    print(f"  Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")
    
    print("\n" + "="*80)
    print("✓ Corpus generation complete!")
    print("="*80)
    
    # Show scaling projection
    print("\n" + "="*80)
    print("SCALING PROJECTION FOR 100K SEQUENCES")
    print("="*80)
    
    tokens_per_seq = stats['avg_tokens_per_doc']
    projected_100k = int(100_000 * tokens_per_seq)
    
    print(f"  Sequences: 100,000")
    print(f"  Tokens per sequence: {tokens_per_seq:.1f}")
    print(f"  Total tokens: {projected_100k:,}")
    print(f"  Estimated size: ~{projected_100k * 4 / 1e9:.2f} GB (int32 tokens)")
    print(f"  Training time (estimate): ~{projected_100k / 1e9 * 10:.1f} hours on 8x A100")


# ============================================================================
# USAGE GUIDE
# ============================================================================

def print_usage_guide():
    """Print usage guide for Qwen integration"""
    
    print("""
USAGE GUIDE: Qwen-Optimized Text Generation
============================================

1. BASIC USAGE
--------------
from hierarchical_event_labeling import HierarchicalEventDataset
from qwen_text_generator import QwenCorpusGenerator

# Create dataset
dataset = HierarchicalEventDataset(your_data)

# Generate corpus (recommended strategy)
generator = QwenCorpusGenerator(strategy='natural_hierarchical')
corpus = generator.generate_corpus(dataset)

# Save
generator.save_corpus(corpus, 'training_corpus.txt')


2. STRATEGY SELECTION
---------------------
Strategy                 Use When                            Tokens/Event
--------                 --------                            ------------
'natural_hierarchical'   Fine-tuning Qwen (RECOMMENDED)     10-15
'natural_flat'           Simple descriptions needed          3-5
'compact'                Maximum efficiency, from scratch    4-6
'hybrid'                 Balance of both                     6-8


3. WITH ACTUAL QWEN TOKENIZER
------------------------------
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B")

# Generate corpus
corpus = generator.generate_corpus(dataset)

# Get actual token counts
stats = generator.estimate_tokens(corpus, tokenizer=tokenizer)
print(f"Actual tokens: {stats['total_tokens']:,}")


4. LARGE-SCALE PROCESSING
--------------------------
# For 100K+ sequences, process in batches
batch_size = 10000

all_corpus = []
for i in range(0, len(all_data), batch_size):
    batch = all_data[i:i+batch_size]
    dataset = HierarchicalEventDataset(batch, verbose=False)
    corpus = generator.generate_corpus(dataset)
    all_corpus.extend(corpus)
    
    # Save intermediate
    generator.save_corpus(corpus, f'corpus_batch_{i}.txt')


5. CUSTOM VOCABULARY MAPPING
-----------------------------
# Extend vocabulary mapping
VocabularyMapper.VOCAB_TO_NATURAL['YOUR_EVENT'] = 'your description'

# Then use normally
corpus = generator.generate_corpus(dataset)
""")


if __name__ == "__main__":
    # Run example
    example_qwen_workflow()
    
    # Print guide
    print_usage_guide()