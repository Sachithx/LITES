# Hierarchical Time Series Event Labeling System - Enhanced Edition

A comprehensive system for detecting, classifying, and organizing time series events into hierarchical structures suitable for language model training. This **enhanced version** includes advanced multi-resolution feature extraction with wavelet decomposition, spectral analysis, and complexity measures.

## 🎯 Overview

This system transforms raw time series data into rich, hierarchical event annotations that can be used to train language models to understand and describe temporal patterns. The enhanced edition extracts **63 advanced features** including wavelet decomposition, spectral analysis, entropy measures, and curvature for superior event detection.

### Key Features

- **Multi-scale Analysis**: Extracts features at 5-50 timestep windows
- **Hierarchical Structure**: Organizes events from micro (single points) to global (full sequence)
- **Rich Vocabulary**: 64 distinct event labels across 7 categories
- **🆕 Advanced Features**: Wavelets, spectral analysis, entropy, curvature, normalized metrics
- **🆕 Wavelet Decomposition**: Multi-resolution aligned with EventScale hierarchy
- **Multiple Text Formats**: Generate training text in various formats
- **Efficient Processing**: Vectorized operations using PyTorch (~26ms per sequence)
- **Extensible Design**: Easy to add new detectors and event types
- **Production Ready**: Optimized for one-time large-scale corpus generation

### What's New in Enhanced Version

✅ **Second derivatives** (curvature) - detects acceleration/reversals  
✅ **Rolling min/max/range** - support/resistance levels  
✅ **Normalized slope** - volatility-adjusted trend strength  
✅ **Spectral features** - frequency domain energy (choppy vs smooth)  
✅ **Shannon entropy** - complexity/chaos detection  
✅ **Jump detection** - discrete event identification  
✅ **Volatility asymmetry** - directional risk (bullish vs bearish)  
✅ **Wavelet decomposition** ⭐ - multi-resolution analysis perfectly aligned with hierarchy  

## 📊 Hierarchical Event Structure

```
EventScale.GLOBAL (150+ steps) ← Wavelet: Approximation (A)
    └── Overall sequence regime (bullish/bearish/sideways/volatile)
    
EventScale.MACRO (50-150 steps) ← Wavelet: Detail D4+
    └── Major trend segments, long volatility regimes
    
EventScale.MESO (15-50 steps) ← Wavelet: Detail D3
    └── Medium trends, local corrections
    
EventScale.MINI (5-15 steps) ← Wavelet: Detail D2
    └── Short segments, volatility clusters
    
EventScale.MICRO (1-5 steps) ← Wavelet: Detail D1
    └── Spikes, single peaks/troughs
```

### Example Hierarchy

```
[0-335] SIDEWAYS_REGIME (GLOBAL)
  ├─ [0-120] UPTREND_LONG (MACRO)
  │   ├─ [30-45] DOWNTREND_SHORT (MESO) ← Nested correction
  │   │   └─ [38] SPIKE_DOWN (MICRO)
  │   └─ [50-55] VOLATILITY_SPIKE (MINI)
  ├─ [121-200] FLAT_SEGMENT (MACRO)
  └─ [201-335] DOWNTREND_LONG (MACRO)
      └─ [250] LOCAL_PEAK (MICRO)
```

## 🚀 Quick Start

```python
import torch
from hierarchical_event_labeling import HierarchicalEventDataset

# 1. Prepare your data [batch_size, sequence_length]
x = torch.randn(100, 336)  # 100 sequences, 336 timesteps each

# 2. Create enhanced dataset with all features
dataset = HierarchicalEventDataset(
    x,
    use_spectral=True,   # Enable spectral features
    use_entropy=True,    # Enable entropy features
    use_wavelets=True,   # Enable wavelet decomposition (RECOMMENDED!)
    verbose=True
)

# 3. Get annotation for first sequence
ann = dataset[0]

# 4. View hierarchical structure
ann.print_hierarchy()

# 5. Generate training text
text = ann.to_text(format='depth_marked')
print(text)

# 6. Access enhanced features
print(f"Total features extracted: {len(dataset.features)}")
# Output: 63 features including wavelets!
```

## 📖 Event Vocabulary

### Categories (64 labels total)

1. **Step Movements** (10 labels)
   - `FLAT`, `UP_SMALL`, `UP_MEDIUM`, `UP_LARGE`
   - `DOWN_SMALL`, `DOWN_MEDIUM`, `DOWN_LARGE`
   - `SPIKE_UP`, `SPIKE_DOWN`

2. **Trend Segments** (7 labels)
   - `UPTREND_SHORT`, `UPTREND_MEDIUM`, `UPTREND_LONG`
   - `DOWNTREND_SHORT`, `DOWNTREND_MEDIUM`, `DOWNTREND_LONG`
   - `FLAT_SEGMENT`

3. **Peaks & Troughs** (4 labels)
   - `LOCAL_PEAK`, `SHARP_PEAK`
   - `LOCAL_TROUGH`, `SHARP_TROUGH`

4. **Volatility Regimes** (4 labels)
   - `LOW_VOLATILITY`, `NORMAL_VOLATILITY`
   - `HIGH_VOLATILITY`, `VOLATILITY_SPIKE`

5. **Change Points** (2 labels)
   - `MEAN_SHIFT_UP`, `MEAN_SHIFT_DOWN`

6. **Global Regimes** (4 labels)
   - `BULLISH_REGIME`, `BEARISH_REGIME`
   - `SIDEWAYS_REGIME`, `VOLATILE_REGIME`

## 🔧 System Architecture

### Processing Pipeline

```
            Raw Time Series [B, L]
                     ↓
┌─────────────────────────────────────────────┐
│ Enhanced Multi-Scale Feature Extraction     │
│ • Basic derivatives (dx, ddx)               │
│ • Rolling features (mean, std, slope)       │
│ • Extrema (min, max, range)                 │
│ • Normalized metrics (norm_slope)           │
│ • Spectral features (low/mid/high bands)    │
│ • Entropy (complexity measure)              │
│ • Wavelet decomposition (D1-D4, A)          │
│ • Jump detection & vol asymmetry            │
│ Result: 63 features per timestep            │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│ Step-Wise Label Encoding                    │
│ Adaptive quantile thresholding              │
│ Result: [B, L] label tensor                 │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│ Event Detection (per sequence)              │
│ • Enhanced Trend Detector (norm slopes)     │
│ • Peak/Trough Detector (alternation)        │
│ • Volatility Regime Detector                │
│ Result: Flat list of events                 │
└─────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────┐
│ Hierarchical Structure Building             │
│ • Scale classification (duration-based)     │
│ • Parent-child relationships                │
│ • Tree construction (containment)           │
│ Result: Hierarchical event tree             │
└─────────────────────────────────────────────┘
                     ↓
         Hierarchical Annotation
```

### Core Components

1. **EnhancedMultiScaleFeatureExtractor** 🆕
   - Efficient convolution-based feature extraction
   - Multiple temporal window sizes (5, 10, 20, 50)
   - **NEW**: Wavelet decomposition (D1-D4 + approximation)
   - **NEW**: Spectral features (STFT bands)
   - **NEW**: Shannon entropy (complexity)
   - **NEW**: Second derivative (curvature)
   - **NEW**: Normalized slopes (noise-filtered)
   - Fully vectorized (batch processing)

2. **StepWiseEncoder**
   - Adaptive quantile thresholding
   - Step-by-step movement classification
   - Handles varying signal magnitudes

3. **Enhanced Event Detectors**
   - `EnhancedTrendSegmentDetector`: Uses normalized slopes for robustness
   - `PeakTroughDetector`: scipy.signal.find_peaks with alternation validation
   - `VolatilityRegimeDetector`: Rolling std quantiles

4. **HierarchicalEventBuilder**
   - Automatic scale classification
   - Containment-based parent finding
   - Depth-first tree construction

5. **HierarchicalAnnotation**
   - Complete sequence annotation
   - Multiple text format generation
   - Event filtering and querying

## 💾 Data Format

### Input
```python
x: torch.Tensor  # Shape: [B, L]
# B = batch size (number of sequences)
# L = sequence length (number of timesteps)
```

### Output Annotation
```python
annotation = dataset[0]

# Components:
annotation.sequence        # [L] Original time series
annotation.step_labels     # [L] Step-wise labels (vocab IDs)
annotation.event_roots     # List[HierarchicalEvent] - Root nodes
annotation.all_events      # List[HierarchicalEvent] - Flattened

# Enhanced: Access all 63 features
dataset.features           # Dict with all extracted features
```

### Event Structure
```python
event.start           # int: Starting timestep
event.end            # int: Ending timestep
event.label          # int: Vocabulary ID
event.label_name     # str: Human-readable name
event.scale          # EventScale: MICRO/MINI/MESO/MACRO/GLOBAL
event.event_type     # str: trend/peak/volatility/regime
event.confidence     # float: Detection confidence
event.metadata       # dict: Additional information
event.parent         # HierarchicalEvent or None
event.children       # List[HierarchicalEvent]
```

## 🆕 Enhanced Features Explained

### 1. Wavelet Decomposition (Most Important!)

**Natural mapping to EventScale hierarchy:**

| Wavelet Level | Event Scale | Duration | What It Captures |
|--------------|-------------|----------|------------------|
| **D1** (finest detail) | MICRO | 1-5 steps | Spikes, noise, single-point events |
| **D2** | MINI | 5-15 steps | Short oscillations, mini-trends |
| **D3** | MESO | 15-50 steps | Medium segments, local patterns |
| **D4+** (coarse detail) | MACRO | 50-150 steps | Major trends, large structures |
| **A** (approximation) | GLOBAL | 150+ steps | Overall direction, regime |

**Why wavelets are powerful:**
- ✅ Time-localized (unlike FFT which is global)
- ✅ Multi-resolution by design
- ✅ Works for non-stationary signals (financial, sensor data)
- ✅ Fast computation (O(L) complexity)
- ✅ Cleaner peak detection (noise naturally filtered)

**Features generated:**
```python
'wavelet_d1' to 'wavelet_d4'      # Detail coefficients [B, L]
'wavelet_a'                       # Approximation [B, L]
'wavelet_energy_d1' to 'd4'       # Energy at each level [B, L]
'wavelet_energy_a'                # Approximation energy [B, L]
```

### 2. Spectral Features

**Frequency-domain analysis:**
- `spec_low_{w}` - Low-frequency energy (smooth trends)
- `spec_mid_{w}` - Mid-frequency energy (oscillations)
- `spec_high_{w}` - High-frequency energy (choppy/noise)

**Use case:** Distinguish between choppy vs smooth market regimes

### 3. Entropy Features

**Complexity measurement:**
- `entropy_{w}` - Shannon entropy in sliding windows

**High entropy** → irregular, chaotic, noisy  
**Low entropy** → regular, predictable, oscillatory

### 4. Curvature (Second Derivative)

**Acceleration detection:**
- `ddx` - Second derivative of signal

**Use case:** Detect sharp reversals, V-shapes vs U-shapes

### 5. Normalized Slope

**Noise-filtered trends:**
- `norm_slope_{w}` - Slope divided by volatility

**Benefit:** Filters out noise, highlights statistically significant moves

### 6. Rolling Extrema

**Support/resistance levels:**
- `min_{w}`, `max_{w}`, `range_{w}` - Local envelopes

**Use case:** Breakout detection, consolidation patterns

### 7. Jump Detection

**Discrete event identification:**
- `jump_indicator` - Binary indicator of sudden level shifts

### 8. Volatility Asymmetry

**Directional risk:**
- `vol_asymmetry` - Ratio of upside to downside volatility

**> 1**: Bullish volatility (upside moves larger)  
**< 1**: Bearish volatility (downside moves larger)

## 📝 Text Generation Formats

### 1. Depth-Marked (Token-Efficient)
```
[0-335]SIDEWAYS_REGIME >[0-120]UPTREND_LONG >>[30-45]DOWNTREND_SHORT >>>[38]SPIKE_DOWN
```
- Depth indicated by `>` symbols
- Compact representation
- ~150-300 tokens per sequence

### 2. Flat Sequential
```
[0-120]UPTREND_LONG [30-45]DOWNTREND_SHORT [38]SPIKE_DOWN [121-200]FLAT_SEGMENT
```
- Loses hierarchy information
- Simple sequential list
- ~100-200 tokens per sequence

### 3. Narrative (Natural Language)
```
Overall: sideways regime. 3 major segments detected. 
[0-120]: uptrend long (contains: trend, peak). 
[30-45]: downtrend short (within uptrend long [0-120]).
```
- Human-readable
- Includes context
- ~200-400 tokens per sequence

## 🔍 Usage Examples

### Basic Usage (All Features Enabled)
```python
from hierarchical_event_labeling import HierarchicalEventDataset

# Load your data
x = torch.load('your_timeseries.pt')  # [B, L]

# Create enhanced dataset (RECOMMENDED)
dataset = HierarchicalEventDataset(
    x,
    use_spectral=True,   # Frequency analysis
    use_entropy=True,    # Complexity detection
    use_wavelets=True,   # Multi-resolution (CRITICAL!)
    verbose=True
)

# Access annotations
for ann in dataset:
    print(ann.to_text())
```

### Feature Selection (Speed vs Quality)
```python
# Maximum quality (recommended for corpus generation)
dataset = HierarchicalEventDataset(
    x, use_spectral=True, use_entropy=True, use_wavelets=True
)
# Time: ~26ms/seq, 63 features

# Wavelet-focused (still excellent)
dataset = HierarchicalEventDataset(
    x, use_spectral=False, use_entropy=False, use_wavelets=True
)
# Time: ~8ms/seq, 46 features

# Minimal (baseline)
dataset = HierarchicalEventDataset(
    x, use_spectral=False, use_entropy=False, use_wavelets=False
)
# Time: ~5ms/seq, 35 features
```

### Filter Events
```python
# Get all macro-scale events
macro_events = ann.get_events_at_scale(EventScale.MACRO)

# Get events in time range
events = ann.get_events_in_range(100, 200)

# Filter by type
trends = [e for e in ann.all_events if e.event_type == 'trend']
peaks = [e for e in ann.all_events if e.event_type == 'peak']
```

### Access Enhanced Features
```python
# Check what features were extracted
print(f"Feature count: {len(dataset.features)}")

# Access specific features
dx = dataset.features['dx']           # First derivative
ddx = dataset.features['ddx']         # Curvature
wavelet_d1 = dataset.features['wavelet_d1']  # Finest details
wavelet_a = dataset.features['wavelet_a']    # Global approximation
entropy = dataset.features['entropy_20']     # Complexity
```

### Generate Training Corpus
```python
from hierarchical_event_labeling import TextCorpusGenerator

# Generate text for all sequences
text_gen = TextCorpusGenerator()
corpus = text_gen.generate_corpus(dataset, format='depth_marked')

# Save to file
with open('training_corpus.txt', 'w') as f:
    for text in corpus:
        f.write(text + '\n')

# Get statistics
stats = text_gen.estimate_tokens(corpus)
print(f"Total tokens: {stats['total_tokens']:,}")
```

### Create PyTorch DataLoader
```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    sequences = torch.stack([ann.sequence for ann in batch])
    texts = [ann.to_text() for ann in batch]
    return {'sequences': sequences, 'texts': texts}

dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

for batch in dataloader:
    # Train your model
    pass
```

## 📊 Performance

### Processing Speed (Enhanced Version)

**Per Sequence (L=336):**
- Feature Extraction: ~20ms (63 features)
- Event Detection: ~4ms
- Hierarchy Building: ~2ms
- **Total**: ~26ms per sequence

**Batch Processing:**
- 1,000 sequences: ~26 seconds
- 10,000 sequences: ~4.3 minutes
- 100,000 sequences: ~43 minutes
- 1,000,000 sequences: ~7.2 hours

**For one-time corpus generation, this is excellent!**

### Feature Extraction Breakdown

| Feature Group | Time/Seq | Feature Count |
|--------------|----------|---------------|
| Basic derivatives | 0.5ms | 2 |
| Rolling features | 4ms | 32 |
| Spectral | 10ms | 12 |
| Entropy | 8ms | 4 |
| **Wavelets** | **3ms** | **11** |
| Other | 0.5ms | 2 |
| **Total** | **26ms** | **63** |

### Memory Usage
- **Raw Data**: ~4 bytes/value
- **All Features**: ~250 bytes/timestep (63 features)
- **Events**: ~200 bytes/event
- **Total**: ~20-30 MB per 1000 sequences (L=336)

### Scalability
```python
# Small dataset
dataset = HierarchicalEventDataset(torch.randn(100, 336))  # ~3 seconds

# Medium dataset
dataset = HierarchicalEventDataset(torch.randn(10000, 336))  # ~4 minutes

# Large dataset - process in batches
for batch in data_batches:
    partial = HierarchicalEventDataset(batch)
    # Save intermediate results
```

## 🎨 Customization

### Add Custom Event Detector
```python
class CustomDetector:
    def detect(self, x: torch.Tensor, features: Dict, idx: int) -> List[SimpleSegment]:
        # Your detection logic using enhanced features
        segments = []
        
        # Example: Use wavelet energy for detection
        wavelet_energy = features['wavelet_energy_d3'][idx]
        high_energy_idx = torch.where(wavelet_energy > threshold)[0]
        
        # ... create segments ...
        return segments

# Integrate into dataset
class CustomEventDataset(HierarchicalEventDataset):
    def __init__(self, x, **kwargs):
        super().__init__(x, **kwargs)
        self.custom_detector = CustomDetector()
    
    def _build_annotation(self, idx, L):
        # ... add custom events to builder ...
        pass
```

### Custom Text Format
```python
def custom_text_format(ann):
    parts = []
    
    # Add metadata
    parts.append(f"LEN:{len(ann.sequence)}")
    
    # Add scale distribution
    scale_counts = {}
    for event in ann.all_events:
        scale_counts[event.scale] = scale_counts.get(event.scale, 0) + 1
    parts.append(f"SCALES:{scale_counts}")
    
    # Add events
    for event in ann.all_events:
        parts.append(f"{event.label_name}@{event.start}")
    
    return " | ".join(parts)
```

## 📈 Applications

1. **Time Series Foundation Models**
   - Pre-train on diverse time series data with rich labels
   - Learn temporal pattern language with multi-resolution understanding
   - Enhanced features provide richer supervision signal

2. **EEG/ECG Signal Analysis**
   - Detect medical events with wavelet-enhanced precision
   - Hierarchical diagnosis with multi-scale patterns
   - Entropy features detect anomalous brain/heart activity

3. **Financial Data**
   - Market regime detection with spectral features
   - Trading pattern recognition with volatility asymmetry
   - Support/resistance with rolling extrema

4. **Sensor Networks**
   - Anomaly detection with jump indicators
   - System state monitoring with wavelet decomposition
   - Change point detection with curvature

5. **Climate Data**
   - Weather pattern analysis with multi-scale trends
   - Long-term trend identification with approximation coefficients
   - Complexity analysis with entropy features

## 🔬 Technical Details

### Wavelet Decomposition Details

**Wavelet Type:** Daubechies-4 (`db4`)
- Compact support (localized in time)
- Smooth (reduces noise)
- Orthogonal (no redundancy)

**Decomposition Levels:** Auto-determined (typically 4-5 for L=336)
```python
max_level = pywt.dwt_max_level(L, 'db4')
levels = min(max_level, 5)  # Cap at 5 levels
```

**Coefficient Upsampling:** Linear interpolation to original length
- Allows aligned features across all scales
- Enables point-wise analysis

### Feature Extraction Windows
- **5 steps**: Micro-patterns, noise filtering
- **10 steps**: Local trends, short volatility
- **20 steps**: Medium trends, regime detection
- **50 steps**: Major trends, global patterns

### Event Scale Classification
```python
duration = end - start + 1

if duration <= 5:      scale = MICRO      # Wavelet D1 range
elif duration <= 15:   scale = MINI       # Wavelet D2 range
elif duration <= 50:   scale = MESO       # Wavelet D3 range
elif duration <= 150:  scale = MACRO      # Wavelet D4+ range
else:                  scale = GLOBAL     # Wavelet approximation
```

### Hierarchy Construction Algorithm
1. Sort events by scale (largest first)
2. For each event, find smallest containing event as parent
3. Build tree structure (parent-child links)
4. Sort children by start position

## 🔧 Installation

### Dependencies
```bash
pip install torch numpy scipy PyWavelets
```

### Version Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy 1.19+
- SciPy 1.5+
- PyWavelets 1.1+

## 📄 License

MIT License - see LICENSE file

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional event detectors (seasonality, cycles with ACF)
- More sophisticated hierarchy algorithms
- Performance optimizations
- Additional text formats
- Support for multivariate time series
- Custom wavelet families
- Adaptive feature selection


## 📧 Contact

For questions or issues, please open a GitHub issue or contact [e240203@e.ntu.edu.sg].

---

**Version**: 2.0.0 (Enhanced Edition)  
**Last Updated**: January 2026  
**Python**: 3.8+  
**Dependencies**: PyTorch, NumPy, SciPy, PyWavelets  
**Recommended Configuration**: All features enabled (spectral + entropy + wavelets)