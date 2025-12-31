
# ============================================================================
# SECTION 5: TOKENIZER FOR EVENT TEXT
# ============================================================================

"""
Event Text Tokenizer with Fixed Position Vocabulary
===================================================

Optimized tokenizer that pre-defines all position markers and structural tokens.
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import json


class EventTextTokenizer:
    """
    Tokenizer optimized for time series event text with fixed position vocabulary.
    
    Special tokens:
        <pad>: Padding token (ID: 0)
        <bos>: Beginning of sequence (ID: 1)
        <eos>: End of sequence (ID: 2)
        <unk>: Unknown token (ID: 3)
        <sep>: Separator between events (ID: 4)
        <mask>: Masked token for pretraining (ID: 5)
    
    Fixed structural tokens:
        [: Left bracket
        ]: Right bracket
        -: Hyphen (for ranges)
        ,: Comma
        0-511: All possible position indices
    
    Event vocabulary:
        Pre-defined event terms + corpus-derived terms
    """
    
    # Special tokens
    PAD_TOKEN = '<pad>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    SEP_TOKEN = '<sep>'
    MASK_TOKEN = '<mask>'
    
    def __init__(self, max_position: int = 512, max_vocab_size: int = 5000):
        """
        Args:
            max_position: Maximum position index (default: 512)
            max_vocab_size: Maximum total vocabulary size
        """
        self.max_position = max_position
        self.max_vocab_size = max_vocab_size
        
        # Initialize vocabulary
        self.vocab = {}
        self.id_to_token = {}
        
        # Build fixed vocabulary
        self._build_fixed_vocab()
        
        # Track variable vocabulary start
        self.variable_vocab_start = len(self.vocab)
    
    def _build_fixed_vocab(self):
        """Build fixed vocabulary: special tokens + structural tokens + positions"""
        
        # 1. Special tokens (0-5)
        special_tokens = [
            self.PAD_TOKEN,  # 0
            self.BOS_TOKEN,  # 1
            self.EOS_TOKEN,  # 2
            self.UNK_TOKEN,  # 3
            self.SEP_TOKEN,  # 4
            self.MASK_TOKEN, # 5
        ]
        
        for token in special_tokens:
            self.vocab[token] = len(self.vocab)
        
        # 2. Structural tokens (6-9)
        structural_tokens = [
            '[',   # 6
            ']',   # 7
            '-',   # 8
            ',',   # 9
        ]
        
        for token in structural_tokens:
            self.vocab[token] = len(self.vocab)
        
        # 3. Position numbers (10 to 10+max_position)
        for i in range(self.max_position):
            self.vocab[str(i)] = len(self.vocab)
        
        # 4. Common metadata tokens
        metadata_tokens = [
            '<sequence',
            'length=',
            'events=',
            '>',
        ]
        
        for token in metadata_tokens:
            self.vocab[token] = len(self.vocab)
        
        # 5. Core event vocabulary
        self.core_event_vocab = [
            # Trends
            'upward', 'downward', 'trend', 'short', 'medium', 'long',
            'flat', 'stable', 'segment',
            
            # Peaks/troughs
            'peak', 'trough', 'local', 'sharp', 'prominent', 'deep',
            
            # Volatility
            'volatility', 'low', 'normal', 'high', 'sudden', 'spike',
            'period',
            
            # Regimes
            'regime', 'bullish', 'bearish', 'sideways', 'consolidation',
            'volatile', 'highly',
            
            # Common words
            'the', 'a', 'an', 'is', 'at', 'in', 'during', 'there',
            'signal', 'exhibits', 'position', 'occurs', 'this',
        ]
        
        for token in self.core_event_vocab:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Update reverse mapping
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Store token IDs for convenience
        self.pad_token_id = self.vocab[self.PAD_TOKEN]
        self.bos_token_id = self.vocab[self.BOS_TOKEN]
        self.eos_token_id = self.vocab[self.EOS_TOKEN]
        self.unk_token_id = self.vocab[self.UNK_TOKEN]
        self.sep_token_id = self.vocab[self.SEP_TOKEN]
        self.mask_token_id = self.vocab[self.MASK_TOKEN]
        
        print(f"Fixed vocabulary initialized: {len(self.vocab)} tokens")
        print(f"  Special tokens: 6")
        print(f"  Structural: 4")
        print(f"  Positions (0-{self.max_position-1}): {self.max_position}")
        print(f"  Core event vocab: {len(self.core_event_vocab)}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into atomic units with proper handling of position markers.
        
        Strategy:
            [0-127] → ['[', '0', '-', '127', ']']
            [23] → ['[', '23', ']']
            <sequence length=128 events=24> → ['<sequence', 'length=', '128', 'events=', '24', '>']
        
        Args:
            text: Raw text string
        
        Returns:
            List of tokens
        """
        tokens = []
        
        # Pattern for metadata: <sequence length=128 events=24>
        metadata_pattern = r'<sequence\s+length=(\d+)\s+events=(\d+)>'
        
        # Pattern for position markers: [0-127] or [23]
        position_pattern = r'\[(\d+)(?:-(\d+))?\]'
        
        i = 0
        while i < len(text):
            # Try to match metadata
            metadata_match = re.match(metadata_pattern, text[i:])
            if metadata_match:
                tokens.extend([
                    '<sequence',
                    'length=',
                    metadata_match.group(1),
                    'events=',
                    metadata_match.group(2),
                    '>'
                ])
                i += metadata_match.end()
                continue
            
            # Try to match position marker
            pos_match = re.match(position_pattern, text[i:])
            if pos_match:
                tokens.append('[')
                tokens.append(pos_match.group(1))  # Start position
                
                if pos_match.group(2):  # Range [start-end]
                    tokens.append('-')
                    tokens.append(pos_match.group(2))  # End position
                
                tokens.append(']')
                i += pos_match.end()
                continue
            
            # Match comma
            if text[i] == ',':
                tokens.append(',')
                i += 1
                continue
            
            # Match whitespace (skip)
            if text[i].isspace():
                i += 1
                continue
            
            # Match word
            word_match = re.match(r'\w+', text[i:])
            if word_match:
                tokens.append(word_match.group(0))
                i += word_match.end()
                continue
            
            # Unknown character (skip)
            i += 1
        
        return tokens
    
    def build_vocab(self, corpus: List[str], min_freq: int = 2):
        """
        Build vocabulary from corpus (adds to fixed vocab).
        
        Args:
            corpus: List of text strings
            min_freq: Minimum frequency for token inclusion
        """
        print(f"\nBuilding vocabulary from {len(corpus)} documents...")
        
        # Count token frequencies (only non-fixed tokens)
        token_freq = Counter()
        for text in corpus:
            tokens = self._tokenize_text(text)
            for token in tokens:
                if token not in self.vocab:  # Only count new tokens
                    token_freq.update([token])
        
        print(f"  Found {len(token_freq)} new unique tokens")
        
        # Add frequent tokens up to max vocab size
        for token, freq in token_freq.most_common():
            if len(self.vocab) >= self.max_vocab_size:
                break
            
            if freq >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Update reverse mapping
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        print(f"  Final vocabulary size: {len(self.vocab)}")
        print(f"    Fixed: {self.variable_vocab_start}")
        print(f"    Variable: {len(self.vocab) - self.variable_vocab_start}")
        print(f"  Coverage: {self._compute_coverage(corpus):.2%}")
        
        return self
    
    def _compute_coverage(self, corpus: List[str]) -> float:
        """Compute vocabulary coverage on corpus"""
        total_tokens = 0
        covered_tokens = 0
        
        for text in corpus[:min(100, len(corpus))]:
            tokens = self._tokenize_text(text)
            total_tokens += len(tokens)
            covered_tokens += sum(1 for t in tokens if t in self.vocab)
        
        return covered_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> Dict[str, List[int]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad to max_length
            truncation: Truncate to max_length
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Tokenize
        tokens = self._tokenize_text(text)
        
        # Convert to IDs
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Add special tokens
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        # Truncate
        if truncation and max_length is not None:
            ids = ids[:max_length]
        
        # Pad
        attention_mask = [1] * len(ids)
        if padding and max_length is not None:
            pad_length = max_length - len(ids)
            if pad_length > 0:
                ids = ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> Dict[str, List[List[int]]]:
        """
        Encode batch of texts.
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        # Encode all texts
        encoded = [self.encode(text, add_special_tokens=add_special_tokens,
                              padding=False, truncation=False)
                  for text in texts]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(enc['input_ids']) for enc in encoded)
        
        # Truncate
        if truncation:
            for enc in encoded:
                enc['input_ids'] = enc['input_ids'][:max_length]
                enc['attention_mask'] = enc['attention_mask'][:max_length]
        
        # Pad
        if padding:
            for enc in encoded:
                pad_length = max_length - len(enc['input_ids'])
                if pad_length > 0:
                    enc['input_ids'].extend([self.pad_token_id] * pad_length)
                    enc['attention_mask'].extend([0] * pad_length)
        
        return {
            'input_ids': [enc['input_ids'] for enc in encoded],
            'attention_mask': [enc['attention_mask'] for enc in encoded]
        }
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Remove special tokens
            clean_up_tokenization: Reconstruct proper formatting
        
        Returns:
            Decoded text string
        """
        tokens = []
        
        for id in ids:
            token = self.id_to_token.get(id, self.UNK_TOKEN)
            
            # Skip special tokens
            if skip_special_tokens and token in [
                self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN,
                self.SEP_TOKEN, self.MASK_TOKEN
            ]:
                continue
            
            tokens.append(token)
        
        # Reconstruct text
        if clean_up_tokenization:
            text = self._reconstruct_text(tokens)
        else:
            text = ' '.join(tokens)
        
        return text
    
    def _reconstruct_text(self, tokens: List[str]) -> str:
        """Reconstruct properly formatted text from tokens"""
        result = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Reconstruct position markers
            if token == '[':
                # Find closing bracket
                j = i + 1
                bracket_tokens = []
                while j < len(tokens) and tokens[j] != ']':
                    bracket_tokens.append(tokens[j])
                    j += 1
                
                # Build position marker
                position_str = '[' + ''.join(bracket_tokens) + ']'
                result.append(position_str)
                i = j + 1
                continue
            
            # Reconstruct metadata
            if token == '<sequence':
                # Collect metadata tokens
                j = i + 1
                meta_tokens = [token]
                while j < len(tokens) and tokens[j] != '>':
                    meta_tokens.append(tokens[j])
                    j += 1
                if j < len(tokens):
                    meta_tokens.append(tokens[j])
                
                # Build metadata string
                meta_str = '<sequence length=' + meta_tokens[2] + ' events=' + meta_tokens[4] + '>'
                result.append(meta_str)
                i = j + 1
                continue
            
            # Handle commas (no space before, space after)
            if token == ',':
                if result:
                    result[-1] = result[-1] + ','
                i += 1
                continue
            
            # Regular token
            result.append(token)
            i += 1
        
        return ' '.join(result)
    
    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode batch of token IDs"""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'max_position': self.max_position,
            'max_vocab_size': self.max_vocab_size,
            'variable_vocab_start': self.variable_vocab_start,
            'core_event_vocab': self.core_event_vocab
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EventTextTokenizer':
        """Load tokenizer from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            max_position=data['max_position'],
            max_vocab_size=data['max_vocab_size']
        )
        
        # Override vocab with loaded one
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.id_to_token = {int(v): k for k, v in data['vocab'].items()}
        tokenizer.variable_vocab_start = data['variable_vocab_start']
        tokenizer.core_event_vocab = data['core_event_vocab']
        
        # Update token IDs
        tokenizer.pad_token_id = tokenizer.vocab[tokenizer.PAD_TOKEN]
        tokenizer.bos_token_id = tokenizer.vocab[tokenizer.BOS_TOKEN]
        tokenizer.eos_token_id = tokenizer.vocab[tokenizer.EOS_TOKEN]
        tokenizer.unk_token_id = tokenizer.vocab[tokenizer.UNK_TOKEN]
        tokenizer.sep_token_id = tokenizer.vocab[tokenizer.SEP_TOKEN]
        tokenizer.mask_token_id = tokenizer.vocab[tokenizer.MASK_TOKEN]
        
        print(f"Tokenizer loaded from {filepath}")
        print(f"  Vocabulary size: {len(tokenizer.vocab)}")
        
        return tokenizer
    
    def __len__(self):
        return len(self.vocab)
    
    def get_vocab_size(self):
        return len(self.vocab)