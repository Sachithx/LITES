"""
Command-line interface for LITES.
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

from lites import EventDetector, EventClassifier, HierarchyBuilder, LLMFormatter
from lites.utils import preprocessing


def load_time_series(filepath: str) -> tuple:
    """Load time series data from file."""
    path = Path(filepath)

    if path.suffix == ".json":
        with open(filepath, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return np.array(data["values"]), np.array(data.get("timestamps", None))
            else:
                return np.array(data), None
    elif path.suffix in [".csv", ".txt"]:
        data = np.loadtxt(filepath, delimiter=",")
        if data.ndim == 2 and data.shape[1] == 2:
            return data[:, 1], data[:, 0]
        return data, None
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LITES: Hierarchical Time Series Event Labeling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Input time series file (JSON, CSV, or TXT)")
    parser.add_argument("-o", "--output", help="Output file path", default=None)
    parser.add_argument(
        "-d",
        "--detection-method",
        choices=["threshold", "changepoint", "peak", "anomaly"],
        default="threshold",
        help="Event detection method",
    )
    parser.add_argument(
        "-c",
        "--classification-method",
        choices=["rule_based", "feature_based"],
        default="rule_based",
        help="Event classification method",
    )
    parser.add_argument(
        "-hs",
        "--hierarchy-strategy",
        choices=["temporal", "class_based", "causal"],
        default="temporal",
        help="Hierarchy building strategy",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "text", "conversation", "instruction"],
        default="json",
        help="Output format for LLM training",
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize time series data"
    )
    parser.add_argument("--smooth", action="store_true", help="Smooth time series data")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for detection (method-specific)",
    )
    parser.add_argument(
        "--time-window", type=float, default=10.0, help="Time window for temporal hierarchy"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    try:
        # Load time series data
        if args.verbose:
            print(f"Loading time series from {args.input}...")
        time_series, timestamps = load_time_series(args.input)

        # Preprocess if requested
        if args.normalize:
            if args.verbose:
                print("Normalizing time series...")
            time_series = preprocessing.normalize_time_series(time_series)

        if args.smooth:
            if args.verbose:
                print("Smoothing time series...")
            time_series = preprocessing.smooth_time_series(time_series)

        # Print statistics
        if args.verbose:
            stats = preprocessing.calculate_statistics(time_series)
            print(f"\nTime series statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        # Detect events
        if args.verbose:
            print(f"\nDetecting events using {args.detection_method} method...")

        detector_params = {}
        if args.threshold is not None:
            detector_params["threshold"] = args.threshold

        detector = EventDetector(method=args.detection_method, **detector_params)
        events = detector.detect(time_series, timestamps)

        if args.verbose:
            print(f"Detected {len(events)} events")

        if not events:
            print("No events detected. Try adjusting detection parameters.")
            return 0

        # Classify events
        if args.verbose:
            print(f"\nClassifying events using {args.classification_method} method...")

        classifier = EventClassifier()
        classified_events = classifier.classify(events, method=args.classification_method)

        if args.verbose:
            print(f"Classified {len(classified_events)} events")
            class_counts = {}
            for ce in classified_events:
                path = ce.get_class_path()
                class_counts[path] = class_counts.get(path, 0) + 1
            print("\nEvent distribution:")
            for path, count in sorted(class_counts.items()):
                print(f"  {path}: {count}")

        # Build hierarchy
        if args.verbose:
            print(f"\nBuilding hierarchy using {args.hierarchy_strategy} strategy...")

        hierarchy_params = {}
        if args.hierarchy_strategy == "temporal":
            hierarchy_params["time_window"] = args.time_window

        builder = HierarchyBuilder(strategy=args.hierarchy_strategy)
        root = builder.build_hierarchy(classified_events, **hierarchy_params)

        if args.verbose:
            print(f"Built hierarchy with {builder._count_nodes(root)} nodes")

        # Format for LLM training
        if args.verbose:
            print(f"\nFormatting output as {args.format}...")

        formatter = LLMFormatter(format_type=args.format)
        formatted_output = formatter.format_hierarchy(root)

        # Save or print output
        if args.output:
            formatter.save_to_file(formatted_output, args.output)
            if args.verbose:
                print(f"\nOutput saved to {args.output}")
        else:
            if isinstance(formatted_output, (dict, list)):
                print(json.dumps(formatted_output, indent=2))
            else:
                print(formatted_output)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
