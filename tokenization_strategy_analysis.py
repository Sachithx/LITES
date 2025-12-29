"""
Tokenization Strategy Analysis for Hierarchical Time Series Events
===================================================================

Comparing different approaches to tokenize event labels with Qwen/LLM tokenizers
for optimal training of multimodal time series foundation models.
"""

import torch
from typing import List, Dict, Tuple
import json


# ============================================================================
# PROBLEM: How to represent hierarchical events for LM training?
# ============================================================================

"""
Key Questions:
1. Should we use raw vocabulary tokens (UP_SMALL, DOWNTREND_LONG)?
2. Should we use natural language descriptions?
3. Should we add special tokens to vocabulary?
4. How to preserve hierarchical structure?

Trade-offs:
- Raw tokens: Compact but may fragment (UP_SMALL → "UP", "_", "SM", "ALL")
- Natural language: Better for pre-trained LMs but verbose
- Special tokens: Efficient but requires vocabulary extension
- Hierarchy: Need depth markers or structured format
"""


# ============================================================================
# STRATEGY 1: Raw Vocabulary Tokens (Current Approach)
# ============================================================================

def strategy_1_raw_tokens():
    """
    Use event labels directly: UP_SMALL, FLAT, SIDEWAYS_REGIME
    
    Pros:
        + Compact representation
        + Clear semantic units
        + Easy to parse back
    
    Cons:
        - May fragment with BPE tokenizer
        - Not natural language (harder for pre-trained models)
        - Underscores may tokenize strangely
    """
    
    example = {
        'format': 'raw_tokens',
        'text': 'UP_SMALL FLAT DOWNTREND_LONG SIDEWAYS_REGIME LOCAL_PEAK',
        'expected_issues': [
            'UP_SMALL might become: ["UP", "_", "SM", "ALL"] (4 tokens!)',
            'SIDEWAYS_REGIME might become: ["SIDE", "WAYS", "_", "REG", "IME"]',
            'Inconsistent tokenization across similar labels'
        ],
        'token_efficiency': 'Low (3-5 tokens per label)',
        'semantic_preservation': 'Medium (fragmented)'
    }
    
    return example


# ============================================================================
# STRATEGY 2: Natural Language Descriptions
# ============================================================================

def strategy_2_natural_language():
    """
    Convert to natural language: "small upward movement", "flat", etc.
    
    Pros:
        + Works well with pre-trained LMs
        + More human-readable
        + Better cross-domain transfer
    
    Cons:
        - Very verbose (3-5x more tokens)
        - Harder to parse back to structured events
        - May lose precision
    """
    
    # Vocabulary mapping
    vocab_to_nl = {
        'UP_SMALL': 'small upward movement',
        'UP_MEDIUM': 'medium upward movement', 
        'UP_LARGE': 'large upward movement',
        'FLAT': 'flat unchanging',
        'DOWNTREND_LONG': 'long downward trend',
        'SIDEWAYS_REGIME': 'sideways consolidation regime',
        'LOCAL_PEAK': 'local peak',
        'SHARP_PEAK': 'sharp prominent peak',
        'BULLISH_REGIME': 'bullish upward regime',
        'VOLATILITY_SPIKE': 'volatility spike',
    }
    
    example = {
        'format': 'natural_language',
        'text': 'small upward movement, flat unchanging, long downward trend, sideways consolidation regime, local peak',
        'vocab_mapping': vocab_to_nl,
        'token_efficiency': 'Low (2-4 tokens per label)',
        'semantic_preservation': 'High',
        'training_benefit': 'Better for pre-trained models'
    }
    
    return example


# ============================================================================
# STRATEGY 3: Hybrid Compressed Format
# ============================================================================

def strategy_3_hybrid_compressed():
    """
    Use abbreviated codes with natural separators
    
    Format: <event_code>[position]
    Examples: <up_sm>[10], <flat>[15-20], <peak>[50]
    
    Pros:
        + Compact (1-2 tokens per event)
        + Consistent tokenization
        + Preserves structure
        + Easy to add to vocabulary
    
    Cons:
        - Requires vocabulary extension
        - Not immediately human-readable
        - Need careful design
    """
    
    # Compressed vocabulary
    compressed_vocab = {
        'UP_SMALL': '<up_sm>',
        'UP_MEDIUM': '<up_md>',
        'UP_LARGE': '<up_lg>',
        'FLAT': '<flat>',
        'DOWNTREND_SHORT': '<dn_sh>',
        'DOWNTREND_MEDIUM': '<dn_md>',
        'DOWNTREND_LONG': '<dn_lg>',
        'SIDEWAYS_REGIME': '<side>',
        'BULLISH_REGIME': '<bull>',
        'BEARISH_REGIME': '<bear>',
        'LOCAL_PEAK': '<pk>',
        'SHARP_PEAK': '<pk_sharp>',
        'LOCAL_TROUGH': '<tr>',
        'VOLATILITY_SPIKE': '<vol_spike>',
    }
    
    example = {
        'format': 'hybrid_compressed',
        'text': '<up_sm>[10] <flat>[15-20] <dn_lg>[25-80] <side>[0-335] <pk>[50]',
        'compressed_vocab': compressed_vocab,
        'token_efficiency': 'High (1-2 tokens per event)',
        'semantic_preservation': 'High',
        'requires_vocab_extension': True,
        'num_new_tokens': len(compressed_vocab)
    }
    
    return example


# ============================================================================
# STRATEGY 4: Hierarchical JSON-style
# ============================================================================

def strategy_4_structured():
    """
    Use JSON-like structured format
    
    Pros:
        + Preserves full hierarchy
        + Machine-parseable
        + Flexible metadata
    
    Cons:
        - Very verbose
        - Many structural tokens
        - May confuse LM
    """
    
    example_text = '''
{
    "global": {"type": "SIDEWAYS_REGIME", "span": [0, 335]},
    "macro": [
        {
            "type": "UPTREND_LONG", 
            "span": [0, 120],
            "children": [
                {"type": "LOCAL_PEAK", "pos": 50}
            ]
        },
        {"type": "DOWNTREND_LONG", "span": [201, 335]}
    ]
}
'''
    
    example = {
        'format': 'structured_json',
        'text': example_text,
        'token_efficiency': 'Very Low (10-20 tokens per event)',
        'semantic_preservation': 'Perfect',
        'hierarchy_preservation': 'Perfect',
        'use_case': 'Complex multi-level analysis'
    }
    
    return example


# ============================================================================
# STRATEGY 5: Depth-Marked Compact (RECOMMENDED)
# ============================================================================

def strategy_5_depth_marked_optimized():
    """
    Optimized depth-marked format with single-token events
    
    Format: >[event]:[span]
    Depth indicated by > count
    Event codes are compact abbreviations
    
    Pros:
        + Very compact (1-3 tokens total per event)
        + Preserves hierarchy via > markers
        + Consistent tokenization
        + Easy to add 64 tokens to vocabulary
    
    Cons:
        - Requires vocabulary extension
        - Need to learn depth marker meaning
    """
    
    # Design compact single-token vocabulary
    compact_vocab = {
        # Movement
        'FLAT': 'FL',
        'UP_SMALL': 'U1',
        'UP_MEDIUM': 'U2', 
        'UP_LARGE': 'U3',
        'DOWN_SMALL': 'D1',
        'DOWN_MEDIUM': 'D2',
        'DOWN_LARGE': 'D3',
        'SPIKE_UP': 'SU',
        'SPIKE_DOWN': 'SD',
        
        # Trends
        'UPTREND_SHORT': 'UT1',
        'UPTREND_MEDIUM': 'UT2',
        'UPTREND_LONG': 'UT3',
        'DOWNTREND_SHORT': 'DT1',
        'DOWNTREND_MEDIUM': 'DT2',
        'DOWNTREND_LONG': 'DT3',
        'FLAT_SEGMENT': 'FLT',
        
        # Peaks
        'LOCAL_PEAK': 'PK',
        'SHARP_PEAK': 'PKS',
        'LOCAL_TROUGH': 'TR',
        'SHARP_TROUGH': 'TRS',
        
        # Volatility
        'LOW_VOLATILITY': 'VL',
        'NORMAL_VOLATILITY': 'VN',
        'HIGH_VOLATILITY': 'VH',
        'VOLATILITY_SPIKE': 'VS',
        
        # Regimes
        'BULLISH_REGIME': 'RB',
        'BEARISH_REGIME': 'RR',
        'SIDEWAYS_REGIME': 'RS',
        'VOLATILE_REGIME': 'RV',
    }
    
    # Example sequences
    examples = {
        'simple': 'RS:0-335 >UT3:0-120 >>PK:50 >>TR:75 >DT3:201-335',
        'complex': 'RS:0-335 >UT3:0-120 >>DT1:30-45 >>>SD:38 >>VH:50-70 >FLT:121-200 >DT3:201-335 >>TR:250',
        
        'tokenization': [
            'Each event: 1 token (e.g., UT3, PK, RS)',
            'Position: 2-4 tokens (e.g., :0-335)',
            'Depth marker: 1 token per level (>, >>)',
            'Total: 4-6 tokens per event'
        ],
        
        'token_efficiency': 'High (4-6 tokens per event)',
        'hierarchy_preservation': 'Perfect (via depth markers)',
        'vocab_extension_size': 64,  # One token per event type
    }
    
    return examples, compact_vocab


# ============================================================================
# STRATEGY 6: Natural Language with Hierarchy (BEST FOR PRE-TRAINED LMs)
# ============================================================================

def strategy_6_natural_hierarchical():
    """
    Natural language descriptions with explicit hierarchy
    
    Format: "[0-335] Overall sideways regime. [0-120] Within this, long upward trend. [50] At position 50, local peak."
    
    Pros:
        + Leverages pre-trained LM knowledge
        + Human-readable
        + No vocabulary extension needed
        + Hierarchy explicit in language
    
    Cons:
        - Most verbose (10-15 tokens per event)
        - Harder to parse back
    """
    
    vocab_to_template = {
        'SIDEWAYS_REGIME': 'overall sideways regime',
        'UPTREND_LONG': 'long upward trend',
        'DOWNTREND_LONG': 'long downward trend',
        'LOCAL_PEAK': 'local peak',
        'SHARP_PEAK': 'sharp prominent peak',
        'VOLATILITY_SPIKE': 'sudden volatility spike',
    }
    
    examples = {
        'simple': '[0-335] Overall sideways regime. [0-120] Long upward trend. [50] Local peak at position 50.',
        
        'nested': '[0-335] Overall sideways regime. [0-120] Within this, long upward trend. [30-45] During the uptrend, short downward correction. [38] At position 38, sharp downward spike. [50-70] Following this, period of high volatility.',
        
        'template_patterns': [
            '[{span}] Overall {global_regime}.',
            '[{span}] Within this, {trend_description}.',
            '[{span}] During the {parent_context}, {local_event}.',
            '[{pos}] At position {pos}, {point_event}.',
        ],
        
        'token_efficiency': 'Low (10-15 tokens per event)',
        'human_readability': 'Excellent',
        'pre_trained_lm_benefit': 'Maximum'
    }
    
    return examples, vocab_to_template


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

def recommend_strategy(use_case: str) -> Dict:
    """
    Recommend tokenization strategy based on use case
    
    Args:
        use_case: 'pretrained_lm', 'from_scratch', 'compact', 'interpretable'
    """
    
    recommendations = {
        'pretrained_lm': {
            'strategy': 'Strategy 6: Natural Language with Hierarchy',
            'reasoning': [
                'Leverages existing language understanding',
                'No vocabulary modification needed',
                'Better zero-shot generalization',
                'Human-interpretable outputs'
            ],
            'implementation': strategy_6_natural_hierarchical,
            'example': '[0-335] Overall sideways regime. [0-120] Long upward trend. [50] Local peak.',
            'expected_tokens_per_event': 10,
            'training_efficiency': 'Medium (more tokens but better initialization)'
        },
        
        'from_scratch': {
            'strategy': 'Strategy 5: Depth-Marked Compact',
            'reasoning': [
                'Most token-efficient',
                'Clean structured representation',
                'Perfect hierarchy preservation',
                'Only 64 new tokens to learn'
            ],
            'implementation': strategy_5_depth_marked_optimized,
            'example': 'RS:0-335 >UT3:0-120 >>PK:50',
            'expected_tokens_per_event': 5,
            'training_efficiency': 'High (fewer tokens, faster training)'
        },
        
        'compact': {
            'strategy': 'Strategy 3: Hybrid Compressed',
            'reasoning': [
                'Good balance of compactness and readability',
                'Single-token events',
                'Easy to parse'
            ],
            'implementation': strategy_3_hybrid_compressed,
            'example': '<up_sm>[10] <flat>[15-20] <dn_lg>[25-80]',
            'expected_tokens_per_event': 4,
            'training_efficiency': 'High'
        },
        
        'interpretable': {
            'strategy': 'Strategy 2: Natural Language',
            'reasoning': [
                'Maximum human readability',
                'Easy to understand outputs',
                'Good for debugging'
            ],
            'implementation': strategy_2_natural_language,
            'example': 'small upward movement, flat unchanging, long downward trend',
            'expected_tokens_per_event': 3,
            'training_efficiency': 'Medium'
        }
    }
    
    return recommendations.get(use_case, recommendations['from_scratch'])


# ============================================================================
# TOKENIZATION ANALYZER
# ============================================================================

def analyze_tokenization(text: str, tokenizer=None) -> Dict:
    """
    Analyze how text will be tokenized
    
    Note: Since we can't load Qwen here, we'll simulate the analysis
    """
    
    if tokenizer is None:
        # Simulate tokenization
        analysis = {
            'warning': 'No tokenizer provided - simulated analysis',
            'text': text,
            'estimated_tokens': len(text.split()),
            'concerns': []
        }
        
        # Check for fragmentation issues
        if '_' in text:
            analysis['concerns'].append('Underscores may cause fragmentation')
        if any(c.isupper() for c in text):
            analysis['concerns'].append('CamelCase/UPPERCASE may tokenize inconsistently')
        if text.count('[') > 5:
            analysis['concerns'].append('Many brackets - consider structured format')
    else:
        # Real tokenization
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        analysis = {
            'text': text,
            'tokens': tokens,
            'num_tokens': len(tokens),
            'token_ids': token_ids,
            'avg_chars_per_token': len(text) / len(tokens) if tokens else 0,
        }
    
    return analysis


# ============================================================================
# PRACTICAL IMPLEMENTATION FOR YOUR USE CASE
# ============================================================================

def generate_training_corpus_qwen_optimized(annotations, strategy='natural_hierarchical'):
    """
    Generate Qwen-optimized training corpus from hierarchical annotations
    
    Args:
        annotations: List of HierarchicalAnnotation objects
        strategy: 'compact', 'natural_hierarchical', 'hybrid'
    
    Returns:
        List of text strings optimized for Qwen tokenizer
    """
    
    corpus = []
    
    if strategy == 'natural_hierarchical':
        # Strategy 6: Best for pre-trained Qwen
        vocab_map = {
            'SIDEWAYS_REGIME': 'sideways consolidation regime',
            'BULLISH_REGIME': 'bullish upward regime',
            'BEARISH_REGIME': 'bearish downward regime',
            'UPTREND_SHORT': 'short upward trend',
            'UPTREND_MEDIUM': 'medium upward trend',
            'UPTREND_LONG': 'long upward trend',
            'DOWNTREND_SHORT': 'short downward trend',
            'DOWNTREND_MEDIUM': 'medium downward trend',
            'DOWNTREND_LONG': 'long downward trend',
            'LOCAL_PEAK': 'local peak',
            'SHARP_PEAK': 'sharp prominent peak',
            'LOCAL_TROUGH': 'local trough',
            'SHARP_TROUGH': 'sharp deep trough',
            'UP_SMALL': 'small rise',
            'UP_MEDIUM': 'moderate rise',
            'UP_LARGE': 'large rise',
            'DOWN_SMALL': 'small decline',
            'DOWN_MEDIUM': 'moderate decline',
            'DOWN_LARGE': 'large decline',
            'FLAT': 'stable',
            'VOLATILITY_SPIKE': 'volatility spike',
            'HIGH_VOLATILITY': 'high volatility period',
        }
        
        for ann in annotations:
            parts = []
            
            # Global regime
            global_events = [e for e in ann.all_events if e.scale.name == 'GLOBAL']
            if global_events:
                e = global_events[0]
                desc = vocab_map.get(e.label_name, e.label_name.lower())
                parts.append(f"[{e.start}-{e.end}] The signal exhibits {desc}.")
            
            # Macro events with context
            macro_events = [e for e in ann.all_events if e.scale.name == 'MACRO']
            for e in macro_events:
                desc = vocab_map.get(e.label_name, e.label_name.lower())
                parts.append(f"[{e.start}-{e.end}] There is a {desc}.")
                
                # Add nested children
                if e.children:
                    child_types = set(vocab_map.get(c.label_name, c.label_name.lower()) 
                                    for c in e.children)
                    parts.append(f"Within this segment: {', '.join(child_types)}.")
            
            corpus.append(" ".join(parts))
    
    elif strategy == 'compact':
        # Strategy 5: Most efficient
        compact_map = {
            'SIDEWAYS_REGIME': 'RS', 'BULLISH_REGIME': 'RB', 
            'UPTREND_LONG': 'UT3', 'DOWNTREND_LONG': 'DT3',
            'LOCAL_PEAK': 'PK', 'SHARP_PEAK': 'PKS',
            # ... (use full mapping from strategy_5)
        }
        
        for ann in annotations:
            def format_event(e, depth=0):
                prefix = '>' * depth
                code = compact_map.get(e.label_name, e.label_name)
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
            
            corpus.append(" ".join(all_parts))
    
    return corpus


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("TOKENIZATION STRATEGY ANALYSIS FOR HIERARCHICAL TIME SERIES EVENTS")
    print("="*80)
    
    # Show all strategies
    strategies = [
        ("Raw Tokens", strategy_1_raw_tokens),
        ("Natural Language", strategy_2_natural_language),
        ("Hybrid Compressed", strategy_3_hybrid_compressed),
        ("Structured JSON", strategy_4_structured),
        ("Depth-Marked Compact", strategy_5_depth_marked_optimized),
        ("Natural Hierarchical", strategy_6_natural_hierarchical),
    ]
    
    for name, func in strategies:
        print(f"\n{'='*80}")
        print(f"STRATEGY: {name}")
        print('='*80)
        
        result = func()
        if isinstance(result, tuple):
            examples, vocab = result
            print(json.dumps(examples, indent=2))
        else:
            print(json.dumps(result, indent=2))
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR YOUR USE CASE")
    print("="*80)
    
    print("\n1. If fine-tuning Qwen (pre-trained LM):")
    rec = recommend_strategy('pretrained_lm')
    print(f"   → {rec['strategy']}")
    print(f"   → Example: {rec['example']}")
    print(f"   → Tokens per event: ~{rec['expected_tokens_per_event']}")
    
    print("\n2. If training from scratch (maximum efficiency):")
    rec = recommend_strategy('from_scratch')
    print(f"   → {rec['strategy']}")
    print(f"   → Example: {rec['example']}")
    print(f"   → Tokens per event: ~{rec['expected_tokens_per_event']}")
    
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    print("""
For your Qwen-based multimodal foundation model:

**Use Strategy 6: Natural Language with Hierarchy**

Rationale:
  1. Qwen is pre-trained on natural language - leverage this!
  2. Better cross-domain transfer and generalization
  3. Human-interpretable outputs (important for debugging)
  4. No vocabulary extension needed
  5. Can incorporate domain knowledge through descriptions

Implementation:
  ```python
  text = "[0-335] The signal exhibits sideways consolidation regime. " \\
         "[0-120] There is a long upward trend. " \\
         "[50] At position 50, local peak."
  ```

Expected performance:
  - 100K sequences → ~300M tokens (vs 150M with compact)
  - Better sample efficiency due to pre-training
  - More robust to domain shift
  - Easier to add new event types
    """)