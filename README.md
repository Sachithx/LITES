# LITES

**Hierarchical Time Series Event Labeling System**

A comprehensive system for detecting, classifying, and organizing time series events into hierarchical structures suitable for language model training.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

LITES (Language model Integrated Time-series Event System) provides a complete pipeline for:

1. **Event Detection**: Identify significant events in time series data using multiple detection methods
2. **Event Classification**: Organize events into hierarchical taxonomies with multi-level classification
3. **Hierarchy Building**: Structure events into meaningful relationships (temporal, causal, or class-based)
4. **LLM Formatting**: Generate training data in various formats for language models

## Features

- **Multiple Detection Methods**
  - Threshold-based detection
  - Change point detection
  - Peak/valley detection
  - Anomaly detection (statistical)

- **Hierarchical Classification**
  - Multi-level event taxonomy
  - Rule-based and feature-based classification
  - Custom taxonomy support

- **Flexible Hierarchy Building**
  - Temporal organization (time windows)
  - Class-based organization
  - Causal relationship inference

- **LLM-Ready Output Formats**
  - Structured JSON
  - Natural language descriptions
  - Conversation format (chat models)
  - Instruction-following format

- **Preprocessing Utilities**
  - Normalization (z-score, min-max, robust)
  - Smoothing (moving average, Savitzky-Golay, exponential)
  - Outlier removal
  - Resampling

## Installation

### From Source

```bash
git clone https://github.com/Sachithx/LITES.git
cd LITES
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With Visualization Support

```bash
pip install -e ".[viz]"
```

## Quick Start

### Python API

```python
import numpy as np
from lites import EventDetector, EventClassifier, HierarchyBuilder, LLMFormatter

# Generate or load your time series data
time_series = np.array([1, 1, 5, 6, 1, 1, 8, 1, 1])
timestamps = np.arange(len(time_series))

# 1. Detect events
detector = EventDetector(method="threshold", threshold=3.0)
events = detector.detect(time_series, timestamps)
print(f"Detected {len(events)} events")

# 2. Classify events
classifier = EventClassifier()
classified_events = classifier.classify(events)

# 3. Build hierarchy
builder = HierarchyBuilder(strategy="temporal")
root = builder.build_hierarchy(classified_events, time_window=10.0)

# 4. Format for LLM training
formatter = LLMFormatter(format_type="json")
llm_data = formatter.format_hierarchy(root)

# Save formatted data
formatter.save_to_file(llm_data, "output.json")
```

### Command-Line Interface

```bash
# Basic usage
lites input_data.json -o output.json

# With specific methods
lites data.csv -d peak -c rule_based -hs temporal -f text

# With preprocessing
lites data.json --normalize --smooth -v

# View all options
lites --help
```

## Usage Examples

### Event Detection Methods

```python
from lites import EventDetector

# Threshold-based detection
detector = EventDetector(method="threshold", threshold=5.0, min_duration=2)
events = detector.detect(time_series)

# Peak detection
detector = EventDetector(method="peak", prominence=2.0, distance=5)
events = detector.detect(time_series)

# Change point detection
detector = EventDetector(method="changepoint", window_size=10, threshold=2.0)
events = detector.detect(time_series)

# Anomaly detection
detector = EventDetector(method="anomaly", z_threshold=3.0)
events = detector.detect(time_series)
```

### Custom Taxonomy

```python
from lites.classification import EventClassifier, EventClass

# Create classifier with custom taxonomy
classifier = EventClassifier()

# Add custom event classes
custom_class = EventClass(
    name="network_spike",
    level=2,
    parent="transient",
    description="Network traffic spike"
)
classifier.add_class(custom_class)

# Classify with custom taxonomy
classified = classifier.classify(events)
```

### Hierarchy Strategies

```python
from lites import HierarchyBuilder

# Temporal hierarchy - group by time windows
builder = HierarchyBuilder(strategy="temporal")
root = builder.build_hierarchy(classified_events, time_window=10.0)

# Class-based hierarchy - group by event classes
builder = HierarchyBuilder(strategy="class_based")
root = builder.build_hierarchy(classified_events)

# Causal hierarchy - infer causal relationships
builder = HierarchyBuilder(strategy="causal")
root = builder.build_hierarchy(classified_events, time_threshold=5.0)

# Visualize hierarchy
print(builder.visualize_hierarchy(root))
```

### LLM Output Formats

```python
from lites import LLMFormatter

# JSON format (structured data)
formatter = LLMFormatter(format_type="json")
json_output = formatter.format_hierarchy(root)

# Text format (natural language)
formatter = LLMFormatter(format_type="text")
text_output = formatter.format_hierarchy(root, context="Network monitoring data")

# Conversation format (for chat models)
formatter = LLMFormatter(format_type="conversation")
conversation = formatter.format_hierarchy(
    root,
    system_prompt="You are a time series expert.",
    query="What patterns do you see?",
    include_response_template=True
)

# Instruction format (for instruction-tuned models)
formatter = LLMFormatter(format_type="instruction")
instruction_data = formatter.format_hierarchy(
    root,
    instruction="Analyze these events and identify patterns."
)
```

### Preprocessing

```python
from lites.utils import preprocessing

# Normalize time series
normalized = preprocessing.normalize_time_series(data, method="zscore")

# Smooth time series
smoothed = preprocessing.smooth_time_series(data, window_size=5, method="moving_average")

# Remove outliers
cleaned, outlier_mask = preprocessing.remove_outliers(data, method="zscore", threshold=3.0)

# Resample
resampled, new_timestamps = preprocessing.resample_time_series(
    data, timestamps, target_length=100
)

# Calculate statistics
stats = preprocessing.calculate_statistics(data)
```

## Architecture

LITES follows a modular pipeline architecture:

```
Time Series Data
      ↓
[Event Detection] ← Multiple methods (threshold, peak, changepoint, anomaly)
      ↓
[Event Classification] ← Hierarchical taxonomy (rule-based or ML-based)
      ↓
[Hierarchy Building] ← Organization strategies (temporal, causal, class-based)
      ↓
[LLM Formatting] ← Multiple output formats (JSON, text, conversation, instruction)
      ↓
Training Data for LLMs
```

## API Reference

### Core Classes

#### `EventDetector`
Detects events in time series data.
- **Methods**: `threshold`, `changepoint`, `peak`, `anomaly`
- **Key Parameters**: method-specific (threshold, window_size, prominence, etc.)

#### `EventClassifier`
Classifies events into hierarchical taxonomies.
- **Methods**: `rule_based`, `feature_based`
- **Features**: Custom taxonomy support, multi-level classification

#### `HierarchyBuilder`
Builds hierarchical structures from classified events.
- **Strategies**: `temporal`, `class_based`, `causal`
- **Features**: Flexible organization, visualization support

#### `LLMFormatter`
Formats hierarchies for language model training.
- **Formats**: `json`, `text`, `conversation`, `instruction`
- **Features**: Batch processing, file I/O

### Event Structure

Events are represented with the following attributes:
- `timestamp`: Time of event occurrence
- `duration`: Event duration
- `magnitude`: Event magnitude/intensity
- `event_type`: Type of event detected
- `confidence`: Detection confidence score
- `metadata`: Additional event-specific data

## Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=lites --cov-report=html

# Run specific test file
pytest tests/test_detection.py -v
```

## Examples

Check the `examples/` directory for complete examples:

```bash
# Run basic example
python examples/basic_usage.py

# This will:
# 1. Generate synthetic time series data
# 2. Detect events using multiple methods
# 3. Classify and organize events
# 4. Generate LLM training data in multiple formats
# 5. Save outputs to examples/output/
```

## Use Cases

- **Anomaly Detection**: Identify and classify unusual patterns in monitoring data
- **Event Analysis**: Structure complex event sequences for analysis
- **LLM Training**: Generate training data for time series understanding models
- **Pattern Recognition**: Discover hierarchical patterns in temporal data
- **Alert Systems**: Build intelligent alerting based on event hierarchies
- **Data Documentation**: Create natural language descriptions of time series events

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use LITES in your research, please cite:

```bibtex
@software{lites2025,
  title={LITES: Hierarchical Time Series Event Labeling System},
  author={Abeywickrama, Sachith},
  year={2025},
  url={https://github.com/Sachithx/LITES}
}
```

## Acknowledgments

Built with:
- NumPy for numerical operations
- SciPy for signal processing
- Scikit-learn for machine learning utilities
- Pandas for data manipulation

## Contact

For questions or feedback, please open an issue on GitHub.
