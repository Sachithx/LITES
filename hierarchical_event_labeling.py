"""
HIERARCHICAL TIME SERIES EVENT LABELING SYSTEM
===============================================

A comprehensive system for detecting and labeling events in time series data
with hierarchical structure preservation.

Workflow:
    1. Define vocabulary and hierarchical structures
    2. Extract multi-scale features from raw time series
    3. Encode step-wise labels for each timestep
    4. Detect higher-level events (trends, peaks, volatility, change points)
    5. Build hierarchical event tree
    6. Generate training text in various formats

Author: Sachith Abeywickrama
Date: December 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from scipy import signal as scipy_signal
from enum import IntEnum
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SECTION 1: CORE DATA STRUCTURES
# ============================================================================

class EventScale(IntEnum):
    """Hierarchical scale levels for events"""
    MICRO = 1      # 1-5 timesteps (spikes, single points)
    MINI = 2       # 5-15 timesteps (very short segments)
    MESO = 3       # 15-50 timesteps (medium segments, local patterns)
    MACRO = 4      # 50-150 timesteps (major trends)
    GLOBAL = 5     # 150+ timesteps (full sequence characteristics)


@dataclass
class EventVocabulary:
    """
    Complete event vocabulary with 64 distinct labels.
    
    Categories:
        - Special tokens (0-2)
        - Step movements (3-10)
        - Trend segments (20-26)
        - Peaks/troughs (30-33)
        - Volatility regimes (40-43)
        - Change points (50-51)
        - Global regimes (60-63)
    """
    # Special tokens
    PAD = 0
    MASK = 1
    FLAT = 2
    
    # Step-level movements
    UP_SMALL = 3
    UP_MEDIUM = 4
    UP_LARGE = 5
    DOWN_SMALL = 6
    DOWN_MEDIUM = 7
    DOWN_LARGE = 8
    SPIKE_UP = 9
    SPIKE_DOWN = 10
    
    # Trend segments
    UPTREND_SHORT = 20
    UPTREND_MEDIUM = 21
    UPTREND_LONG = 22
    DOWNTREND_SHORT = 23
    DOWNTREND_MEDIUM = 24
    DOWNTREND_LONG = 25
    FLAT_SEGMENT = 26
    
    # Peaks and troughs
    LOCAL_PEAK = 30
    SHARP_PEAK = 31
    LOCAL_TROUGH = 32
    SHARP_TROUGH = 33
    
    # Volatility regimes
    LOW_VOLATILITY = 40
    NORMAL_VOLATILITY = 41
    HIGH_VOLATILITY = 42
    VOLATILITY_SPIKE = 43
    
    # Change points
    MEAN_SHIFT_UP = 50
    MEAN_SHIFT_DOWN = 51
    
    # Global regimes
    BULLISH_REGIME = 60
    BEARISH_REGIME = 61
    SIDEWAYS_REGIME = 62
    VOLATILE_REGIME = 63
    
    @classmethod
    def get_vocab_size(cls) -> int:
        """Return total vocabulary size"""
        return 64
    
    @classmethod
    def id_to_label(cls, idx: int) -> str:
        """Convert label ID to string name"""
        for name, value in vars(cls).items():
            if isinstance(value, int) and value == idx:
                return name
        return "UNKNOWN"


@dataclass
class HierarchicalEvent:
    """
    Event node in hierarchical tree structure.
    
    Attributes:
        start: Starting timestep index
        end: Ending timestep index
        label: Vocabulary ID
        label_name: Human-readable label
        scale: Hierarchical scale level
        event_type: Category (trend/peak/volatility/changepoint/regime)
        confidence: Detection confidence score
        metadata: Additional event-specific information
        parent: Parent event in hierarchy (None for root)
        children: List of child events
    """
    start: int
    end: int
    label: int
    label_name: str
    scale: EventScale
    event_type: str
    confidence: float
    metadata: Dict
    parent: Optional['HierarchicalEvent'] = None
    children: List['HierarchicalEvent'] = field(default_factory=list)
    
    @property
    def duration(self) -> int:
        """Duration in timesteps"""
        return self.end - self.start + 1
    
    @property
    def depth(self) -> int:
        """Depth in hierarchy tree (0 = root)"""
        depth = 0
        node = self.parent
        while node is not None:
            depth += 1
            node = node.parent
        return depth
    
    def contains(self, other: 'HierarchicalEvent') -> bool:
        """Check if this event fully contains another event"""
        return (self.start <= other.start and 
                self.end >= other.end and
                self != other)
    
    def __repr__(self):
        indent = "  " * self.depth
        children_info = f" ({len(self.children)} children)" if self.children else ""
        return (f"{indent}[{self.start:03d}-{self.end:03d}] {self.label_name} "
                f"(scale={self.scale.name}){children_info}")


# Global vocabulary instance
VOCAB = EventVocabulary()


# ============================================================================
# SECTION 2: FEATURE EXTRACTION
# ============================================================================

class MultiScaleFeatureExtractor:
    """
    Extract features at multiple temporal scales using efficient convolutions.
    
    Features computed:
        - First derivative (dx)
        - Rolling mean at multiple window sizes
        - Rolling standard deviation (volatility)
        - Rolling slope (trend strength)
        - Z-scores for anomaly detection
    
    Args:
        scales: List of window sizes for rolling features
    """
    
    def __init__(self, scales: List[int] = [5, 10, 20, 50]):
        self.scales = scales
    
    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from time series batch.
        
        Args:
            x: Time series tensor of shape [B, L]
        
        Returns:
            Dictionary of feature tensors, each shape [B, L]:
                - 'dx': First derivative
                - 'mean_{w}': Rolling mean with window w
                - 'std_{w}': Rolling std with window w
                - 'slope_{w}': Rolling slope with window w
                - 'zscore': Normalized z-scores
        """
        B, L = x.shape
        device = x.device
        features = {}
        
        # First derivative (rate of change)
        dx = torch.diff(x, dim=1)  # [B, L-1]
        features['dx'] = F.pad(dx, (1, 0), value=0)  # [B, L]
        
        # Multi-scale rolling features
        for w in self.scales:
            if w >= L:
                continue
            
            x_3d = x.unsqueeze(1)  # [B, 1, L]
            kernel = torch.ones(1, 1, w, device=device) / w
            padding = w - 1
            
            # Rolling mean using efficient convolution
            x_padded = F.pad(x_3d, (padding, 0), mode='replicate')
            rolling_mean = F.conv1d(x_padded, kernel).squeeze(1)
            features[f'mean_{w}'] = rolling_mean
            
            # Rolling standard deviation (volatility measure)
            x_diff = x.unsqueeze(1) - rolling_mean.unsqueeze(1)
            x_diff_padded = F.pad(x_diff, (padding, 0), mode='replicate')
            rolling_var = F.conv1d(x_diff_padded ** 2, kernel).squeeze(1)
            features[f'std_{w}'] = torch.sqrt(rolling_var.clamp(min=1e-8))
            
            # Rolling slope (trend direction and strength)
            slopes = self._compute_slopes(x, w)
            features[f'slope_{w}'] = slopes
        
        # Z-scores for outlier detection
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp(min=1e-8)
        features['zscore'] = (x - mean) / std
        
        return features
    
    def _compute_slopes(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """
        Compute rolling linear slope over window.
        
        Simple implementation: slope = (end - start) / window
        """
        B, L = x.shape
        slopes = torch.zeros(B, L, device=x.device)
        
        for i in range(window, L):
            slopes[:, i] = (x[:, i] - x[:, i-window+1]) / window
        
        return slopes


# ============================================================================
# SECTION 3: STEP-WISE LABEL ENCODING
# ============================================================================

class StepWiseEncoder:
    """
    Encode each timestep with symbolic movement labels.
    
    Labels based on magnitude of first derivative:
        - FLAT: negligible change
        - UP/DOWN_SMALL/MEDIUM/LARGE: quantile-based magnitude bins
        - SPIKE_UP/DOWN: extreme changes (>90th percentile)
    """
    
    def encode(self, x: torch.Tensor, features: Dict) -> torch.Tensor:
        """
        Encode step-wise movement labels for entire batch.
        
        Args:
            x: Time series tensor [B, L]
            features: Feature dictionary from MultiScaleFeatureExtractor
        
        Returns:
            Label tensor [B, L] with vocabulary IDs
        """
        B, L = x.shape
        device = x.device
        
        dx = features['dx']
        abs_dx = torch.abs(dx[:, 1:])  # Skip first padded value
        
        if abs_dx.numel() == 0:
            return torch.full((B, L), VOCAB.FLAT, dtype=torch.long, device=device)
        
        # Compute quantiles for adaptive thresholding
        q33, q66, q90 = torch.quantile(
            abs_dx.reshape(-1),
            torch.tensor([0.33, 0.66, 0.90], device=device)
        )
        
        epsilon = 0.1 * q33  # Flat threshold
        labels = torch.full((B, L), VOCAB.PAD, dtype=torch.long, device=device)
        
        # Classify each timestep
        for t in range(1, L):
            diff = dx[:, t]
            abs_diff = torch.abs(diff)
            
            # Flat (negligible change)
            flat_mask = abs_diff < epsilon
            labels[flat_mask, t] = VOCAB.FLAT
            
            # Upward movements
            up_mask = (diff > 0) & (~flat_mask)
            labels[up_mask & (abs_diff <= q33), t] = VOCAB.UP_SMALL
            labels[up_mask & (abs_diff > q33) & (abs_diff <= q66), t] = VOCAB.UP_MEDIUM
            labels[up_mask & (abs_diff > q66) & (abs_diff <= q90), t] = VOCAB.UP_LARGE
            labels[up_mask & (abs_diff > q90), t] = VOCAB.SPIKE_UP
            
            # Downward movements
            down_mask = (diff < 0) & (~flat_mask)
            labels[down_mask & (abs_diff <= q33), t] = VOCAB.DOWN_SMALL
            labels[down_mask & (abs_diff > q33) & (abs_diff <= q66), t] = VOCAB.DOWN_MEDIUM
            labels[down_mask & (abs_diff > q66) & (abs_diff <= q90), t] = VOCAB.DOWN_LARGE
            labels[down_mask & (abs_diff > q90), t] = VOCAB.SPIKE_DOWN
        
        return labels


# ============================================================================
# SECTION 4: EVENT DETECTORS
# ============================================================================

@dataclass
class SimpleSegment:
    """Simple segment representation for detector outputs"""
    start: int
    end: int
    label: int
    metadata: Dict = field(default_factory=dict)


class TrendSegmentDetector:
    """
    Detect trend segments using slope sign changes.
    
    Algorithm:
        1. Compute rolling slopes
        2. Find sign changes (trend reversals)
        3. Classify segments by direction and duration
    """
    
    def detect(self, x: torch.Tensor, features: Dict, idx: int) -> List[SimpleSegment]:
        """
        Detect trend segments in single sequence.
        
        Args:
            x: Single sequence tensor [L]
            features: Feature dictionary
            idx: Sequence index in batch
        
        Returns:
            List of trend segments
        """
        x_np = x.cpu().numpy()
        L = len(x_np)
        
        # Get slopes
        if 'slope_20' in features:
            slopes = features['slope_20'][idx].cpu().numpy()
        else:
            slopes = np.gradient(x_np)
        
        # Find trend direction changes
        slope_sign = np.sign(slopes)
        slope_sign[np.abs(slopes) < 0.01] = 0  # Flat threshold
        
        changes = np.where(np.diff(slope_sign) != 0)[0] + 1
        breakpoints = np.concatenate([[0], changes, [L]])
        
        segments = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1] - 1
            
            if end - start < 5:  # Skip very short segments
                continue
            
            avg_slope = slopes[start:end+1].mean()
            duration = end - start + 1
            
            # Classify segment
            if abs(avg_slope) < 0.01:
                label = VOCAB.FLAT_SEGMENT
            elif avg_slope > 0:
                if duration < 30:
                    label = VOCAB.UPTREND_SHORT
                elif duration < 80:
                    label = VOCAB.UPTREND_MEDIUM
                else:
                    label = VOCAB.UPTREND_LONG
            else:
                if duration < 30:
                    label = VOCAB.DOWNTREND_SHORT
                elif duration < 80:
                    label = VOCAB.DOWNTREND_MEDIUM
                else:
                    label = VOCAB.DOWNTREND_LONG
            
            segments.append(SimpleSegment(
                start=start,
                end=end,
                label=label,
                metadata={'slope': float(avg_slope)}
            ))
        
        return segments


class PeakTroughDetector:
    """
    Detect peaks and troughs using scipy's find_peaks with proper filtering.
    
    Key improvements:
        - Minimum distance enforcement (prevent adjacent peaks)
        - Adaptive prominence thresholds
        - Peak-trough alternation validation
    
    Classifies peaks/troughs by:
        - Prominence: How much the peak stands out
        - Type: Sharp vs broad based on width
    """
    
    def __init__(self, min_distance: int = 10, min_prominence_percentile: float = 75):
        """
        Args:
            min_distance: Minimum timesteps between peaks/troughs
            min_prominence_percentile: Percentile for adaptive prominence threshold
        """
        self.min_distance = min_distance
        self.min_prominence_percentile = min_prominence_percentile
    
    def detect(self, x: torch.Tensor, idx: int) -> List[SimpleSegment]:
        """
        Detect peaks and troughs in single sequence.
        
        Args:
            x: Single sequence tensor [L]
            idx: Sequence index (unused but kept for consistency)
        
        Returns:
            List of peak/trough events (alternating peaks and troughs)
        """
        x_np = x.cpu().numpy()
        
        # Compute adaptive prominence threshold
        std = np.std(x_np)
        min_prominence = max(0.2 * std, 0.1)  # At least 0.2 std or 0.1 absolute
        
        events = []
        
        # Find peaks
        try:
            peaks, props = scipy_signal.find_peaks(
                x_np, 
                prominence=min_prominence,
                distance=self.min_distance,  # Enforce minimum separation
                width=1
            )
            
            for pk, prom in zip(peaks, props['prominences']):
                # Classify by prominence
                if prom > std:
                    label = VOCAB.SHARP_PEAK
                else:
                    label = VOCAB.LOCAL_PEAK
                
                events.append(SimpleSegment(
                    start=int(pk),
                    end=int(pk),
                    label=label,
                    metadata={
                        'prominence': float(prom),
                        'type': 'peak'
                    }
                ))
        except Exception as e:
            pass
        
        # Find troughs (peaks of inverted signal)
        try:
            troughs, props = scipy_signal.find_peaks(
                -x_np,
                prominence=min_prominence,
                distance=self.min_distance,  # Enforce minimum separation
                width=1
            )
            
            for tr, prom in zip(troughs, props['prominences']):
                # Classify by prominence
                if prom > std:
                    label = VOCAB.SHARP_TROUGH
                else:
                    label = VOCAB.LOCAL_TROUGH
                
                events.append(SimpleSegment(
                    start=int(tr),
                    end=int(tr),
                    label=label,
                    metadata={
                        'prominence': float(prom),
                        'type': 'trough'
                    }
                ))
        except Exception as e:
            pass
        
        # Sort by position and validate alternation
        events = self._validate_alternation(events)
        
        return events
    
    def _validate_alternation(self, events: List[SimpleSegment]) -> List[SimpleSegment]:
        """
        Ensure peaks and troughs alternate (remove consecutive same-type events).
        Keep the more prominent one when there are consecutive same types.
        """
        if len(events) <= 1:
            return events
        
        # Sort by position
        events.sort(key=lambda e: e.start)
        
        filtered = [events[0]]
        
        for event in events[1:]:
            last_event = filtered[-1]
            
            # Check if types alternate
            last_type = last_event.metadata.get('type')
            curr_type = event.metadata.get('type')
            
            if last_type == curr_type:
                # Same type consecutive - keep more prominent
                if event.metadata['prominence'] > last_event.metadata['prominence']:
                    filtered[-1] = event  # Replace with more prominent
                # else: keep the existing one
            else:
                # Different types - check minimum distance
                if event.start - last_event.start >= self.min_distance // 2:
                    filtered.append(event)
                # else: skip (too close even if different types)
        
        return filtered


class VolatilityRegimeDetector:
    """
    Detect volatility regimes using rolling standard deviation.
    
    Classifies regimes by quantile thresholds:
        - LOW: Below 25th percentile
        - NORMAL: 25th-75th percentile
        - HIGH: Above 75th percentile
        - SPIKE: Above 90th percentile
    """
    
    def detect(self, x: torch.Tensor, features: Dict, idx: int) -> List[SimpleSegment]:
        """
        Detect volatility regimes in single sequence.
        
        Args:
            x: Single sequence tensor [L]
            features: Feature dictionary
            idx: Sequence index in batch
        
        Returns:
            List of volatility regime segments
        """
        if 'std_20' not in features:
            return []
        
        vol = features['std_20'][idx].cpu().numpy()
        L = len(vol)
        
        # Compute quantile thresholds
        q25, q75, q90 = np.percentile(vol, [25, 75, 90])
        
        # Classify each point
        vol_levels = np.zeros(L, dtype=int)
        vol_levels[vol <= q25] = 0  # low
        vol_levels[(vol > q25) & (vol <= q75)] = 1  # normal
        vol_levels[(vol > q75) & (vol <= q90)] = 2  # high
        vol_levels[vol > q90] = 3  # spike
        
        # Find regime changes
        changes = np.where(np.diff(vol_levels) != 0)[0] + 1
        breakpoints = np.concatenate([[0], changes, [L]])
        
        regimes = []
        for i in range(len(breakpoints) - 1):
            start = breakpoints[i]
            end = breakpoints[i + 1] - 1
            
            if end - start < 5:  # Skip very short regimes
                continue
            
            level_code = vol_levels[start]
            avg_vol = vol[start:end+1].mean()
            
            # Map to vocabulary
            label_map = {
                0: VOCAB.LOW_VOLATILITY,
                1: VOCAB.NORMAL_VOLATILITY,
                2: VOCAB.HIGH_VOLATILITY,
                3: VOCAB.VOLATILITY_SPIKE
            }
            
            regimes.append(SimpleSegment(
                start=start,
                end=end,
                label=label_map[level_code],
                metadata={'avg_volatility': float(avg_vol)}
            ))
        
        return regimes


# ============================================================================
# SECTION 5: HIERARCHICAL STRUCTURE BUILDER
# ============================================================================

class HierarchicalEventBuilder:
    """
    Build hierarchical event tree from flat event list.
    
    Process:
        1. Classify each event's scale based on duration
        2. Sort events by scale (largest first)
        3. Build parent-child relationships via containment
        4. Sort children by temporal order
    """
    
    def __init__(self):
        self.events: List[HierarchicalEvent] = []
    
    def add_event(self, start: int, end: int, label: int, event_type: str,
                  confidence: float = 1.0, metadata: Optional[Dict] = None):
        """
        Add event to collection with automatic scale classification.
        
        Args:
            start: Starting timestep
            end: Ending timestep
            label: Vocabulary ID
            event_type: Category string
            confidence: Detection confidence score
            metadata: Additional information dictionary
        """
        duration = end - start + 1
        
        # Determine hierarchical scale
        if duration <= 5:
            scale = EventScale.MICRO
        elif duration <= 15:
            scale = EventScale.MINI
        elif duration <= 50:
            scale = EventScale.MESO
        elif duration <= 150:
            scale = EventScale.MACRO
        else:
            scale = EventScale.GLOBAL
        
        event = HierarchicalEvent(
            start=start, end=end, label=label,
            label_name=VOCAB.id_to_label(label),
            scale=scale, event_type=event_type,
            confidence=confidence, metadata=metadata or {}
        )
        self.events.append(event)
    
    def build_hierarchy(self) -> List[HierarchicalEvent]:
        """
        Build hierarchical tree structure.
        
        Returns:
            List of root events (events with no parent)
        """
        # Sort by scale (largest first), then by start position
        sorted_events = sorted(self.events, key=lambda e: (-e.scale, e.start))
        roots = []
        
        # Build parent-child relationships
        for event in sorted_events:
            parent = self._find_parent(event, sorted_events)
            if parent is None:
                roots.append(event)
            else:
                event.parent = parent
                parent.children.append(event)
        
        # Sort children within each parent
        self._sort_children(roots)
        return roots
    
    def _find_parent(self, event: HierarchicalEvent,
                     all_events: List[HierarchicalEvent]) -> Optional[HierarchicalEvent]:
        """Find smallest event that contains this event (most specific parent)"""
        candidates = [e for e in all_events if e != event and 
                     e.scale > event.scale and e.contains(event)]
        return min(candidates, key=lambda e: e.duration) if candidates else None
    
    def _sort_children(self, nodes: List[HierarchicalEvent]):
        """Recursively sort children by start position"""
        for node in nodes:
            if node.children:
                node.children.sort(key=lambda c: c.start)
                self._sort_children(node.children)
    
    def get_flat_list(self, roots: List[HierarchicalEvent]) -> List[HierarchicalEvent]:
        """Get flattened list via depth-first traversal"""
        result = []
        def traverse(node):
            result.append(node)
            for child in node.children:
                traverse(child)
        for root in roots:
            traverse(root)
        return result


# ============================================================================
# SECTION 6: HIERARCHICAL ANNOTATION
# ============================================================================

@dataclass
class HierarchicalAnnotation:
    """
    Complete hierarchical annotation for one sequence.
    
    Attributes:
        sequence: Original time series [L]
        step_labels: Step-wise labels [L]
        event_roots: Root nodes of hierarchy tree
        all_events: Flattened list of all events
    """
    sequence: torch.Tensor
    step_labels: torch.Tensor
    event_roots: List[HierarchicalEvent]
    all_events: List[HierarchicalEvent]
    
    def print_hierarchy(self, max_depth: int = 10):
        """Pretty print hierarchical structure"""
        def print_tree(node: HierarchicalEvent, depth: int = 0):
            if depth > max_depth:
                return
            print(node)
            for child in node.children:
                print_tree(child, depth + 1)
        
        print(f"\nHierarchical Events (Total: {len(self.all_events)})")
        print("=" * 80)
        for root in self.event_roots:
            print_tree(root)
    
    def get_events_at_scale(self, scale: EventScale) -> List[HierarchicalEvent]:
        """Get all events at specific hierarchical scale"""
        return [e for e in self.all_events if e.scale == scale]
    
    def get_events_in_range(self, start: int, end: int) -> List[HierarchicalEvent]:
        """Get all events overlapping with time range"""
        return [e for e in self.all_events 
                if not (e.end < start or e.start > end)]
    
    def to_text(self, format: str = 'depth_marked') -> str:
        """
        Generate text representation.
        
        Args:
            format: Output format
                - 'depth_marked': Depth indicators with events
                - 'flat': Simple sequential list
                - 'narrative': Natural language description
        
        Returns:
            Text string for language model training
        """
        if format == 'depth_marked':
            return self._depth_marked_text()
        elif format == 'flat':
            return self._flat_text()
        elif format == 'narrative':
            return self._narrative_text()
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _depth_marked_text(self) -> str:
        """Depth markers indicate nesting: > >> >>> etc."""
        parts = []
        def traverse(node):
            depth_marker = ">" * node.depth
            parts.append(f"{depth_marker}[{node.start}-{node.end}]{node.label_name}")
            for child in node.children:
                traverse(child)
        for root in self.event_roots:
            traverse(root)
        return " ".join(parts)
    
    def _flat_text(self) -> str:
        """Simple sequential list (loses hierarchy)"""
        events = sorted(self.all_events, key=lambda e: e.start)
        return " ".join(f"[{e.start}-{e.end}]{e.label_name}" for e in events)
    
    def _narrative_text(self) -> str:
        """Natural language with hierarchical context"""
        sentences = []
        
        # Start with global view
        global_events = self.get_events_at_scale(EventScale.GLOBAL)
        if global_events:
            sentences.append(
                f"Overall: {global_events[0].label_name.lower().replace('_', ' ')}."
            )
        
        # Describe macro events
        macro_events = self.get_events_at_scale(EventScale.MACRO)
        if macro_events:
            sentences.append(f"{len(macro_events)} major segments detected.")
            for event in macro_events[:3]:
                desc = event.label_name.lower().replace('_', ' ')
                sentences.append(f"[{event.start}-{event.end}]: {desc}")
                if event.children:
                    nested = ", ".join(set(c.event_type for c in event.children))
                    sentences.append(f"  (contains: {nested})")
        
        return " ".join(sentences)


# ============================================================================
# SECTION 7: MAIN DATASET CLASS
# ============================================================================

class HierarchicalEventDataset(Dataset):
    """
    Main dataset class for hierarchical event labeling.
    
    Processing pipeline:
        1. Extract multi-scale features
        2. Encode step-wise labels
        3. Detect events (trends, peaks, volatility)
        4. Add global regime classification
        5. Build hierarchical structure
        6. Create annotations
    
    Args:
        x: Time series tensor [B, L]
        verbose: Print progress messages
    """
    
    def __init__(self, x: torch.Tensor, verbose: bool = True):
        super().__init__()
        
        if x.dim() != 2:
            raise ValueError("Expected x with shape [B, L]")
        
        self.x = x
        B, L = x.shape
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"INITIALIZING HIERARCHICAL EVENT DATASET")
            print(f"{'='*80}")
            print(f"Sequences: {B}")
            print(f"Length: {L}")
        
        # Initialize components
        self.feature_extractor = MultiScaleFeatureExtractor()
        self.step_encoder = StepWiseEncoder()
        self.trend_detector = TrendSegmentDetector()
        self.peak_detector = PeakTroughDetector()
        self.vol_detector = VolatilityRegimeDetector()
        
        # STEP 1: Extract features
        if verbose:
            print(f"\n[1/4] Extracting multi-scale features...")
        self.features = self.feature_extractor.extract_features(x)
        if verbose:
            print(f"      ✓ Computed {len(self.features)} feature types")
        
        # STEP 2: Encode step labels
        if verbose:
            print(f"[2/4] Encoding step-wise labels...")
        self.step_labels = self.step_encoder.encode(x, self.features)
        if verbose:
            print(f"      ✓ Encoded {B * L} timesteps")
        
        # STEP 3: Detect events and build hierarchy
        if verbose:
            print(f"[3/4] Detecting events and building hierarchy...")
        
        self.annotations = []
        for i in range(B):
            if verbose and i % 50 == 0:
                print(f"      Processing sequence {i}/{B}...")
            
            annotation = self._build_annotation(i, L)
            self.annotations.append(annotation)
        
        # STEP 4: Compute statistics
        if verbose:
            print(f"[4/4] Computing statistics...")
            self._print_statistics()
            print(f"\n{'='*80}")
            print(f"✓ DATASET READY")
            print(f"{'='*80}\n")
    
    def _build_annotation(self, idx: int, L: int) -> HierarchicalAnnotation:
        """Build complete hierarchical annotation for one sequence"""
        
        builder = HierarchicalEventBuilder()
        
        # Detect all event types
        trends = self.trend_detector.detect(self.x[idx], self.features, idx)
        peaks = self.peak_detector.detect(self.x[idx], idx)
        vol_regimes = self.vol_detector.detect(self.x[idx], self.features, idx)
        
        # Add trend segments
        for seg in trends:
            builder.add_event(seg.start, seg.end, seg.label, 'trend',
                            confidence=0.9, metadata=seg.metadata)
        
        # Add peaks/troughs
        for pk in peaks:
            builder.add_event(pk.start, pk.end, pk.label, 'peak',
                            confidence=0.85, metadata=pk.metadata)
        
        # Add volatility regimes
        for vr in vol_regimes:
            builder.add_event(vr.start, vr.end, vr.label, 'volatility',
                            confidence=0.8, metadata=vr.metadata)
        
        # Add global regime
        global_label = self._classify_global_regime(idx)
        builder.add_event(0, L-1, global_label, 'regime', confidence=0.7)
        
        # Build hierarchy
        roots = builder.build_hierarchy()
        all_events = builder.get_flat_list(roots)
        
        return HierarchicalAnnotation(
            sequence=self.x[idx],
            step_labels=self.step_labels[idx],
            event_roots=roots,
            all_events=all_events
        )
    
    def _classify_global_regime(self, idx: int) -> int:
        """Classify overall sequence regime"""
        if 'slope_20' in self.features:
            avg_slope = self.features['slope_20'][idx].mean().item()
        else:
            avg_slope = 0
        
        if avg_slope > 0.05:
            return VOCAB.BULLISH_REGIME
        elif avg_slope < -0.05:
            return VOCAB.BEARISH_REGIME
        else:
            return VOCAB.SIDEWAYS_REGIME
    
    def _print_statistics(self):
        """Print dataset statistics"""
        total_events = sum(len(a.all_events) for a in self.annotations)
        avg_events = total_events / len(self.annotations)
        
        # Count by scale
        scale_counts = {scale: 0 for scale in EventScale}
        for ann in self.annotations:
            for event in ann.all_events:
                scale_counts[event.scale] += 1
        
        print(f"      Total events: {total_events}")
        print(f"      Avg per sequence: {avg_events:.1f}")
        print(f"      By scale:")
        for scale in EventScale:
            count = scale_counts[scale]
            avg = count / len(self.annotations)
            print(f"        {scale.name:.<12} {avg:>6.1f} per sequence")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        return self.annotations[idx]
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        total_events = sum(len(a.all_events) for a in self.annotations)
        
        return {
            'num_sequences': len(self.annotations),
            'sequence_length': len(self.annotations[0].sequence),
            'vocab_size': VOCAB.get_vocab_size(),
            'total_events': total_events,
            'avg_events_per_sequence': total_events / len(self.annotations),
        }


# ============================================================================
# SECTION 8: TEXT GENERATION FOR LM TRAINING
# ============================================================================

class TextCorpusGenerator:
    """
    Generate training text in various formats.
    
    Formats:
        - depth_marked: Hierarchical with depth indicators (>)
        - flat: Simple sequential list
        - narrative: Natural language description
    """
    
    @staticmethod
    def generate_corpus(dataset: HierarchicalEventDataset, 
                       format: str = 'depth_marked') -> List[str]:
        """
        Generate text corpus for all sequences.
        
        Args:
            dataset: HierarchicalEventDataset instance
            format: Text format
        
        Returns:
            List of text strings, one per sequence
        """
        corpus = []
        for annotation in dataset.annotations:
            text = annotation.to_text(format=format)
            corpus.append(text)
        return corpus
    
    @staticmethod
    def estimate_tokens(corpus: List[str]) -> Dict:
        """Estimate token counts for corpus"""
        total_tokens = sum(len(text.split()) for text in corpus)
        total_chars = sum(len(text) for text in corpus)
        
        return {
            'num_documents': len(corpus),
            'total_tokens': total_tokens,
            'total_chars': total_chars,
            'avg_tokens_per_doc': total_tokens / len(corpus),
            'avg_chars_per_doc': total_chars / len(corpus),
        }


# ============================================================================
# SECTION 9: DEMONSTRATION & TESTING
# ============================================================================

def generate_synthetic_data(B: int = 50, L: int = 336, seed: int = 42) -> torch.Tensor:
    """
    Generate realistic synthetic time series.
    
    Components:
        - Multi-scale sinusoidal trends
        - Volatility clusters
        - Random spikes
        - Local corrections (creates nested events)
    
    Args:
        B: Batch size (number of sequences)
        L: Sequence length
        seed: Random seed
    
    Returns:
        Tensor of shape [B, L]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    t = torch.linspace(0, 4*np.pi, L)
    x = torch.zeros(B, L)
    
    for i in range(B):
        # Base trend (multiple scales)
        trend = 0.5 * torch.sin(t / 2) + 0.1 * t
        
        # Add volatility clusters
        vol_modulator = 0.1 + 0.2 * (torch.sin(3 * t) > 0).float()
        noise = torch.randn(L) * vol_modulator
        
        # Add random spikes
        num_spikes = np.random.randint(2, 5)
        spike_indices = torch.randint(50, L-50, (num_spikes,))
        spikes = torch.zeros(L)
        spikes[spike_indices] = torch.randn(num_spikes) * 2
        
        # Add local correction (creates nested structure)
        if i % 3 == 0:
            # Dip in middle of uptrend
            start = L // 2
            end = start + 30
            x[i, start:end] = x[i, start:end] - 0.5
        
        x[i] = trend + noise + spikes
    
    return x


def demonstrate_system():
    """Run complete demonstration of the system"""
    
    print("\n" + "="*80)
    print("HIERARCHICAL TIME SERIES EVENT LABELING SYSTEM")
    print("Demonstration")
    print("="*80)
    
    # Generate data
    B, L = 20, 336
    print(f"\nGenerating {B} synthetic sequences of length {L}...")
    x = generate_synthetic_data(B, L)
    
    # Create dataset
    dataset = HierarchicalEventDataset(x, verbose=True)
    
    # Show example annotation
    print("\n" + "="*80)
    print("EXAMPLE: HIERARCHICAL STRUCTURE")
    print("="*80)
    
    ann = dataset[0]
    ann.print_hierarchy(max_depth=3)
    
    # Show events by scale
    print("\n" + "="*80)
    print("EVENTS BY HIERARCHICAL SCALE")
    print("="*80)
    
    for scale in EventScale:
        events = ann.get_events_at_scale(scale)
        print(f"\n{scale.name} ({len(events)} events):")
        for e in events[:5]:
            print(f"  [{e.start:03d}-{e.end:03d}] {e.label_name}")
    
    # Generate text in different formats
    print("\n" + "="*80)
    print("TEXT GENERATION FOR LM TRAINING")
    print("="*80)
    
    formats = ['depth_marked', 'flat', 'narrative']
    for fmt in formats:
        text = ann.to_text(format=fmt)
        tokens = len(text.split())
        chars = len(text)
        print(f"\n{fmt.upper()}:")
        print(f"  Tokens: {tokens}, Chars: {chars}")
        print(f"  Preview: {text[:200]}...")
    
    # Generate full corpus
    print("\n" + "="*80)
    print("FULL CORPUS STATISTICS")
    print("="*80)
    
    text_gen = TextCorpusGenerator()
    corpus = text_gen.generate_corpus(dataset, format='depth_marked')
    stats = text_gen.estimate_tokens(corpus)
    
    for key, value in stats.items():
        print(f"  {key}: {value:,.1f}" if isinstance(value, float) else f"  {key}: {value:,}")
    
    print("\n" + "="*80)
    print("✓ DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe system is ready for processing real time series data.")
    print("Simply create a dataset with: dataset = HierarchicalEventDataset(your_data)")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    demonstrate_system()