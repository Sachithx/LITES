"""
Example: Basic usage of LITES for time series event detection and classification.
"""

import numpy as np
import json
from pathlib import Path

from lites import EventDetector, EventClassifier, HierarchyBuilder, LLMFormatter
from lites.utils import preprocessing


def generate_synthetic_time_series(length=200, seed=42):
    """Generate synthetic time series with various events."""
    np.random.seed(seed)

    # Base signal
    t = np.linspace(0, 10, length)
    signal = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.random.randn(length)

    # Add some spikes
    spike_indices = [50, 100, 150]
    for idx in spike_indices:
        signal[idx] += 5

    # Add a step change
    signal[120:] += 2

    # Add a sustained elevated period
    signal[80:95] += 3

    timestamps = t
    return signal, timestamps


def main():
    """Run example workflow."""
    print("=" * 70)
    print("LITES: Hierarchical Time Series Event Labeling System")
    print("=" * 70)
    print()

    # Generate synthetic data
    print("1. Generating synthetic time series data...")
    time_series, timestamps = generate_synthetic_time_series()
    print(f"   Generated {len(time_series)} data points")
    print()

    # Calculate statistics
    stats = preprocessing.calculate_statistics(time_series)
    print("2. Time series statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    print()

    # Detect events using multiple methods
    print("3. Detecting events...")
    all_events = []

    # Threshold detection
    threshold_detector = EventDetector(method="threshold", threshold=3.0)
    threshold_events = threshold_detector.detect(time_series, timestamps)
    all_events.extend(threshold_events)
    print(f"   Threshold method: {len(threshold_events)} events")

    # Peak detection
    peak_detector = EventDetector(method="peak", prominence=2.0)
    peak_events = peak_detector.detect(time_series, timestamps)
    all_events.extend(peak_events)
    print(f"   Peak method: {len(peak_events)} events")

    # Anomaly detection
    anomaly_detector = EventDetector(method="anomaly", z_threshold=2.5)
    anomaly_events = anomaly_detector.detect(time_series, timestamps)
    all_events.extend(anomaly_events)
    print(f"   Anomaly method: {len(anomaly_events)} events")

    print(f"   Total events detected: {len(all_events)}")
    print()

    # Classify events
    print("4. Classifying events...")
    classifier = EventClassifier()
    classified_events = classifier.classify(all_events, method="rule_based")

    # Count events by class
    class_counts = {}
    for ce in classified_events:
        path = ce.get_class_path()
        class_counts[path] = class_counts.get(path, 0) + 1

    print("   Event distribution by class:")
    for path, count in sorted(class_counts.items()):
        print(f"   - {path}: {count}")
    print()

    # Build hierarchies with different strategies
    print("5. Building event hierarchies...")

    # Temporal hierarchy
    temporal_builder = HierarchyBuilder(strategy="temporal")
    temporal_root = temporal_builder.build_hierarchy(classified_events, time_window=2.0)
    print(f"   Temporal hierarchy: {temporal_builder._count_nodes(temporal_root)} nodes")

    # Class-based hierarchy
    class_builder = HierarchyBuilder(strategy="class_based")
    class_root = class_builder.build_hierarchy(classified_events)
    print(f"   Class-based hierarchy: {class_builder._count_nodes(class_root)} nodes")

    # Causal hierarchy
    causal_builder = HierarchyBuilder(strategy="causal")
    causal_root = causal_builder.build_hierarchy(classified_events, time_threshold=1.0)
    print(f"   Causal hierarchy: {causal_builder._count_nodes(causal_root)} nodes")
    print()

    # Format for LLM training
    print("6. Formatting for LLM training...")

    # JSON format
    json_formatter = LLMFormatter(format_type="json")
    json_output = json_formatter.format_hierarchy(temporal_root)
    print("   ✓ JSON format")

    # Text format
    text_formatter = LLMFormatter(format_type="text")
    text_output = text_formatter.format_hierarchy(temporal_root)
    print("   ✓ Text format")

    # Conversation format
    conv_formatter = LLMFormatter(format_type="conversation")
    conv_output = conv_formatter.format_hierarchy(
        temporal_root, include_response_template=True
    )
    print("   ✓ Conversation format")

    # Instruction format
    inst_formatter = LLMFormatter(format_type="instruction")
    inst_output = inst_formatter.format_hierarchy(temporal_root)
    print("   ✓ Instruction format")
    print()

    # Save outputs
    print("7. Saving outputs...")
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

    # Save JSON
    with open(output_dir / "hierarchy_json.json", "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"   Saved JSON to {output_dir / 'hierarchy_json.json'}")

    # Save text
    with open(output_dir / "hierarchy_text.txt", "w") as f:
        f.write(text_output)
    print(f"   Saved text to {output_dir / 'hierarchy_text.txt'}")

    # Save conversation
    with open(output_dir / "hierarchy_conversation.json", "w") as f:
        json.dump(conv_output, f, indent=2)
    print(f"   Saved conversation to {output_dir / 'hierarchy_conversation.json'}")

    # Save instruction
    with open(output_dir / "hierarchy_instruction.json", "w") as f:
        json.dump(inst_output, f, indent=2)
    print(f"   Saved instruction to {output_dir / 'hierarchy_instruction.json'}")

    # Save input data
    with open(output_dir / "input_data.json", "w") as f:
        json.dump(
            {"timestamps": timestamps.tolist(), "values": time_series.tolist()},
            f,
            indent=2,
        )
    print(f"   Saved input data to {output_dir / 'input_data.json'}")
    print()

    # Display sample text output
    print("8. Sample text output:")
    print("-" * 70)
    print(text_output[:500] + "..." if len(text_output) > 500 else text_output)
    print("-" * 70)
    print()

    print("✓ Example completed successfully!")
    print()
    print("Next steps:")
    print("  - Check the examples/output/ directory for generated files")
    print("  - Try the CLI tool: lites examples/output/input_data.json")
    print("  - Explore different detection methods and parameters")


if __name__ == "__main__":
    main()
