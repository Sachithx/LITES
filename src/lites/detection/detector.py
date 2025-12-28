"""
Event detection module for time series data.

Provides multiple methods for detecting events in time series data including
threshold-based, change point detection, and pattern-based approaches.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import signal
from scipy.stats import zscore


class Event:
    """Represents a detected event in time series data."""

    def __init__(
        self,
        timestamp: float,
        duration: float,
        magnitude: float,
        event_type: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.timestamp = timestamp
        self.duration = duration
        self.magnitude = magnitude
        self.event_type = event_type
        self.confidence = confidence
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "magnitude": self.magnitude,
            "event_type": self.event_type,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"Event(timestamp={self.timestamp}, duration={self.duration}, "
            f"type={self.event_type}, magnitude={self.magnitude:.2f})"
        )


class EventDetector:
    """
    Detects events in time series data using various methods.

    Supports multiple detection strategies:
    - Threshold-based: Detect when values exceed a threshold
    - Change point: Detect significant changes in statistical properties
    - Peak detection: Identify local maxima/minima
    - Anomaly detection: Identify unusual patterns
    """

    def __init__(self, method: str = "threshold", **kwargs):
        """
        Initialize event detector.

        Args:
            method: Detection method ('threshold', 'changepoint', 'peak', 'anomaly')
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.params = kwargs

    def detect(
        self, time_series: np.ndarray, timestamps: Optional[np.ndarray] = None
    ) -> List[Event]:
        """
        Detect events in time series data.

        Args:
            time_series: 1D array of time series values
            timestamps: Optional array of timestamps (defaults to indices)

        Returns:
            List of detected Event objects
        """
        if timestamps is None:
            timestamps = np.arange(len(time_series))

        if self.method == "threshold":
            return self._detect_threshold(time_series, timestamps)
        elif self.method == "changepoint":
            return self._detect_changepoint(time_series, timestamps)
        elif self.method == "peak":
            return self._detect_peaks(time_series, timestamps)
        elif self.method == "anomaly":
            return self._detect_anomaly(time_series, timestamps)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

    def _detect_threshold(
        self, time_series: np.ndarray, timestamps: np.ndarray
    ) -> List[Event]:
        """Detect events where values exceed a threshold."""
        threshold = self.params.get("threshold", np.mean(time_series) + 2 * np.std(time_series))
        min_duration = self.params.get("min_duration", 1)

        events = []
        in_event = False
        event_start = None
        event_values = []

        for i, (ts, value) in enumerate(zip(timestamps, time_series)):
            if value > threshold and not in_event:
                in_event = True
                event_start = i
                event_values = [value]
            elif value > threshold and in_event:
                event_values.append(value)
            elif value <= threshold and in_event:
                duration = i - event_start
                if duration >= min_duration:
                    events.append(
                        Event(
                            timestamp=timestamps[event_start],
                            duration=duration,
                            magnitude=np.max(event_values),
                            event_type="threshold_exceeded",
                            confidence=1.0,
                        )
                    )
                in_event = False
                event_start = None
                event_values = []

        # Handle case where event extends to end of series
        if in_event and len(event_values) >= min_duration:
            duration = len(timestamps) - event_start
            events.append(
                Event(
                    timestamp=timestamps[event_start],
                    duration=duration,
                    magnitude=np.max(event_values),
                    event_type="threshold_exceeded",
                    confidence=1.0,
                )
            )

        return events

    def _detect_changepoint(
        self, time_series: np.ndarray, timestamps: np.ndarray
    ) -> List[Event]:
        """Detect change points using simple statistical approach."""
        window_size = self.params.get("window_size", 10)
        threshold = self.params.get("threshold", 2.0)

        events = []
        if len(time_series) < 2 * window_size:
            return events

        for i in range(window_size, len(time_series) - window_size):
            before = time_series[i - window_size : i]
            after = time_series[i : i + window_size]

            mean_diff = abs(np.mean(after) - np.mean(before))
            std_diff = np.std(before) + 1e-10

            if mean_diff / std_diff > threshold:
                events.append(
                    Event(
                        timestamp=timestamps[i],
                        duration=1,
                        magnitude=mean_diff,
                        event_type="changepoint",
                        confidence=min(mean_diff / std_diff / threshold, 1.0),
                    )
                )

        return events

    def _detect_peaks(
        self, time_series: np.ndarray, timestamps: np.ndarray
    ) -> List[Event]:
        """Detect peaks in time series."""
        prominence = self.params.get("prominence", None)
        distance = self.params.get("distance", 1)

        peaks, properties = signal.find_peaks(
            time_series, prominence=prominence, distance=distance
        )

        events = []
        for peak_idx in peaks:
            events.append(
                Event(
                    timestamp=timestamps[peak_idx],
                    duration=1,
                    magnitude=time_series[peak_idx],
                    event_type="peak",
                    confidence=1.0,
                    metadata={"index": int(peak_idx)},
                )
            )

        return events

    def _detect_anomaly(
        self, time_series: np.ndarray, timestamps: np.ndarray
    ) -> List[Event]:
        """Detect anomalies using z-score method."""
        z_threshold = self.params.get("z_threshold", 3.0)

        z_scores = np.abs(zscore(time_series, nan_policy="omit"))
        anomaly_indices = np.where(z_scores > z_threshold)[0]

        events = []
        for idx in anomaly_indices:
            events.append(
                Event(
                    timestamp=timestamps[idx],
                    duration=1,
                    magnitude=abs(time_series[idx]),
                    event_type="anomaly",
                    confidence=min(z_scores[idx] / z_threshold, 1.0),
                    metadata={"z_score": float(z_scores[idx])},
                )
            )

        return events
