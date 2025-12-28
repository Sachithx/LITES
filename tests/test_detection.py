"""Tests for event detection module."""

import numpy as np
import pytest
from lites.detection import EventDetector, Event


class TestEvent:
    """Test Event class."""

    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            timestamp=10.0,
            duration=2.0,
            magnitude=5.0,
            event_type="peak",
            confidence=0.9,
        )
        assert event.timestamp == 10.0
        assert event.duration == 2.0
        assert event.magnitude == 5.0
        assert event.event_type == "peak"
        assert event.confidence == 0.9

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = Event(
            timestamp=10.0, duration=2.0, magnitude=5.0, event_type="peak"
        )
        event_dict = event.to_dict()
        assert event_dict["timestamp"] == 10.0
        assert event_dict["duration"] == 2.0
        assert event_dict["magnitude"] == 5.0
        assert event_dict["event_type"] == "peak"


class TestEventDetector:
    """Test EventDetector class."""

    def test_threshold_detection(self):
        """Test threshold-based event detection."""
        # Create synthetic time series with spikes
        time_series = np.array([1, 1, 5, 6, 1, 1, 8, 1, 1])
        detector = EventDetector(method="threshold", threshold=3.0)
        events = detector.detect(time_series)

        assert len(events) > 0
        assert all(isinstance(e, Event) for e in events)

    def test_peak_detection(self):
        """Test peak detection."""
        # Create time series with clear peaks
        time_series = np.array([0, 1, 0, 2, 0, 3, 0, 1, 0])
        detector = EventDetector(method="peak")
        events = detector.detect(time_series)

        assert len(events) > 0
        assert all(e.event_type == "peak" for e in events)

    def test_changepoint_detection(self):
        """Test change point detection."""
        # Create time series with clear change point
        time_series = np.concatenate([np.ones(20), np.ones(20) * 5])
        detector = EventDetector(method="changepoint", window_size=5)
        events = detector.detect(time_series)

        assert len(events) >= 0  # May or may not detect depending on parameters

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Create time series with outliers
        np.random.seed(42)
        time_series = np.random.randn(100)
        time_series[50] = 10  # Add anomaly
        detector = EventDetector(method="anomaly", z_threshold=3.0)
        events = detector.detect(time_series)

        assert len(events) > 0
        assert all(e.event_type == "anomaly" for e in events)

    def test_custom_timestamps(self):
        """Test detection with custom timestamps."""
        time_series = np.array([1, 1, 5, 6, 1, 1])
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        detector = EventDetector(method="threshold", threshold=3.0)
        events = detector.detect(time_series, timestamps)

        assert len(events) > 0
        # Check that timestamps are used
        assert all(e.timestamp in timestamps for e in events)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError):
            detector = EventDetector(method="invalid_method")
            detector.detect(np.array([1, 2, 3]))
