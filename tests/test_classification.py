"""Tests for event classification module."""

import numpy as np
import pytest
from lites.detection import EventDetector, Event
from lites.classification import EventClassifier, ClassifiedEvent, EventClass


class TestEventClass:
    """Test EventClass."""

    def test_event_class_creation(self):
        """Test creating an event class."""
        event_class = EventClass(
            name="spike", level=2, parent="transient", description="Sharp increase"
        )
        assert event_class.name == "spike"
        assert event_class.level == 2
        assert event_class.parent == "transient"

    def test_event_class_to_dict(self):
        """Test converting event class to dictionary."""
        event_class = EventClass(name="spike", level=2, parent="transient")
        class_dict = event_class.to_dict()
        assert class_dict["name"] == "spike"
        assert class_dict["level"] == 2


class TestClassifiedEvent:
    """Test ClassifiedEvent."""

    def test_classified_event_creation(self):
        """Test creating a classified event."""
        event = Event(10.0, 1.0, 5.0, "peak")
        classified = ClassifiedEvent(event, ["event", "transient", "spike"])
        assert classified.event == event
        assert classified.classes == ["event", "transient", "spike"]

    def test_get_class_path(self):
        """Test getting class path as string."""
        event = Event(10.0, 1.0, 5.0, "peak")
        classified = ClassifiedEvent(event, ["event", "transient", "spike"])
        assert classified.get_class_path() == "event > transient > spike"


class TestEventClassifier:
    """Test EventClassifier."""

    def test_default_taxonomy(self):
        """Test that default taxonomy is created."""
        classifier = EventClassifier()
        assert len(classifier.taxonomy) > 0
        assert "event" in classifier.taxonomy

    def test_rule_based_classification(self):
        """Test rule-based classification."""
        classifier = EventClassifier()
        events = [
            Event(10.0, 1.0, 5.0, "peak", confidence=1.0),
            Event(20.0, 10.0, 3.0, "threshold_exceeded", confidence=1.0),
        ]
        classified = classifier.classify(events, method="rule_based")

        assert len(classified) == 2
        assert all(isinstance(ce, ClassifiedEvent) for ce in classified)
        assert all(len(ce.classes) > 0 for ce in classified)

    def test_add_class(self):
        """Test adding a custom class to taxonomy."""
        classifier = EventClassifier()
        initial_count = len(classifier.taxonomy)

        new_class = EventClass("custom", 1, "event", "Custom event type")
        classifier.add_class(new_class)

        assert len(classifier.taxonomy) == initial_count + 1
        assert "custom" in classifier.taxonomy

    def test_get_class_hierarchy(self):
        """Test getting class hierarchy path."""
        classifier = EventClassifier()
        hierarchy = classifier.get_class_hierarchy("spike")

        assert isinstance(hierarchy, list)
        assert len(hierarchy) > 0
        assert hierarchy[0] == "event"  # Root should be first

    def test_get_taxonomy_tree(self):
        """Test getting taxonomy as tree structure."""
        classifier = EventClassifier()
        tree = classifier.get_taxonomy_tree()

        assert isinstance(tree, dict)
        assert "event" in tree

    def test_feature_based_classification(self):
        """Test feature-based classification (fallback to rule-based)."""
        classifier = EventClassifier()
        events = [Event(10.0, 1.0, 5.0, "peak", confidence=1.0)]
        classified = classifier.classify(events, method="feature_based")

        assert len(classified) == 1
        assert isinstance(classified[0], ClassifiedEvent)
