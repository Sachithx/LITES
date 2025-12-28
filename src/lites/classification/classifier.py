"""
Event classification module.

Provides hierarchical classification of detected events into multi-level taxonomies.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from ..detection.detector import Event


class EventClass:
    """Represents a class in the event taxonomy."""

    def __init__(
        self,
        name: str,
        level: int,
        parent: Optional[str] = None,
        description: str = "",
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.level = level
        self.parent = parent
        self.description = description
        self.attributes = attributes or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "level": self.level,
            "parent": self.parent,
            "description": self.description,
            "attributes": self.attributes,
        }


class ClassifiedEvent:
    """An event with classification information."""

    def __init__(
        self,
        event: Event,
        classes: List[str],
        confidence_scores: Optional[Dict[str, float]] = None,
    ):
        self.event = event
        self.classes = classes  # Hierarchical list from root to leaf
        self.confidence_scores = confidence_scores or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event": self.event.to_dict(),
            "classes": self.classes,
            "confidence_scores": self.confidence_scores,
        }

    def get_class_path(self) -> str:
        """Get the full hierarchical class path as a string."""
        return " > ".join(self.classes)

    def __repr__(self) -> str:
        return f"ClassifiedEvent(event={self.event}, classes={self.get_class_path()})"


class EventClassifier:
    """
    Classifies events into hierarchical taxonomies.

    Supports both rule-based and feature-based classification methods.
    """

    def __init__(self, taxonomy: Optional[Dict[str, EventClass]] = None):
        """
        Initialize event classifier.

        Args:
            taxonomy: Dictionary mapping class names to EventClass objects
        """
        self.taxonomy = taxonomy or self._create_default_taxonomy()
        self.scaler = StandardScaler()

    def _create_default_taxonomy(self) -> Dict[str, EventClass]:
        """Create a default event taxonomy."""
        taxonomy = {
            "event": EventClass("event", 0, None, "Root event class"),
            # Level 1: Event nature
            "transient": EventClass("transient", 1, "event", "Short-lived events"),
            "sustained": EventClass("sustained", 1, "event", "Long-duration events"),
            "periodic": EventClass("periodic", 1, "event", "Repeating events"),
            # Level 2: Event characteristics
            "spike": EventClass("spike", 2, "transient", "Sudden sharp increase"),
            "dip": EventClass("dip", 2, "transient", "Sudden sharp decrease"),
            "step_up": EventClass("step_up", 2, "sustained", "Sustained increase"),
            "step_down": EventClass("step_down", 2, "sustained", "Sustained decrease"),
            "oscillation": EventClass("oscillation", 2, "periodic", "Regular oscillating pattern"),
            # Level 3: Event severity
            "minor": EventClass("minor", 3, None, "Low impact event"),
            "moderate": EventClass("moderate", 3, None, "Medium impact event"),
            "major": EventClass("major", 3, None, "High impact event"),
        }
        return taxonomy

    def classify(
        self, events: List[Event], method: str = "rule_based"
    ) -> List[ClassifiedEvent]:
        """
        Classify a list of events.

        Args:
            events: List of Event objects to classify
            method: Classification method ('rule_based' or 'feature_based')

        Returns:
            List of ClassifiedEvent objects
        """
        if method == "rule_based":
            return [self._classify_rule_based(event) for event in events]
        elif method == "feature_based":
            return self._classify_feature_based(events)
        else:
            raise ValueError(f"Unknown classification method: {method}")

    def _classify_rule_based(self, event: Event) -> ClassifiedEvent:
        """Classify event using rule-based approach."""
        classes = ["event"]
        confidence_scores = {}

        # Level 1: Determine event nature based on duration
        if event.duration == 1:
            classes.append("transient")
            confidence_scores["transient"] = 1.0
        elif event.duration > 5:
            classes.append("sustained")
            confidence_scores["sustained"] = 1.0
        else:
            # Check if it's part of a periodic pattern (simplified)
            classes.append("transient")
            confidence_scores["transient"] = 0.8

        # Level 2: Determine characteristics
        if event.event_type == "peak":
            if classes[-1] == "transient":
                classes.append("spike")
                confidence_scores["spike"] = event.confidence
            elif classes[-1] == "sustained":
                classes.append("step_up")
                confidence_scores["step_up"] = event.confidence
        elif event.event_type == "anomaly":
            if classes[-1] == "transient":
                classes.append("spike")
                confidence_scores["spike"] = event.confidence
        elif event.event_type == "threshold_exceeded":
            if classes[-1] == "sustained":
                classes.append("step_up")
                confidence_scores["step_up"] = event.confidence
            else:
                classes.append("spike")
                confidence_scores["spike"] = event.confidence

        # Level 3: Determine severity based on magnitude
        magnitude = event.magnitude
        if magnitude < 1.0:
            classes.append("minor")
            confidence_scores["minor"] = 1.0
        elif magnitude < 3.0:
            classes.append("moderate")
            confidence_scores["moderate"] = 1.0
        else:
            classes.append("major")
            confidence_scores["major"] = 1.0

        return ClassifiedEvent(event, classes, confidence_scores)

    def _classify_feature_based(self, events: List[Event]) -> List[ClassifiedEvent]:
        """Classify events using feature-based approach (simplified)."""
        # Extract features
        features = self._extract_features(events)

        # For now, use rule-based as fallback
        # In a full implementation, this would use ML models
        return [self._classify_rule_based(event) for event in events]

    def _extract_features(self, events: List[Event]) -> np.ndarray:
        """Extract features from events for ML classification."""
        features = []
        for event in events:
            feature_vec = [
                event.duration,
                event.magnitude,
                event.confidence,
                1 if event.event_type == "peak" else 0,
                1 if event.event_type == "anomaly" else 0,
                1 if event.event_type == "changepoint" else 0,
            ]
            features.append(feature_vec)
        return np.array(features)

    def add_class(self, event_class: EventClass) -> None:
        """Add a new class to the taxonomy."""
        self.taxonomy[event_class.name] = event_class

    def get_class_hierarchy(self, class_name: str) -> List[str]:
        """Get the full hierarchy path for a class."""
        hierarchy = []
        current = class_name

        while current is not None:
            if current in self.taxonomy:
                hierarchy.insert(0, current)
                current = self.taxonomy[current].parent
            else:
                break

        return hierarchy

    def get_taxonomy_tree(self) -> Dict[str, Any]:
        """Get the taxonomy as a nested dictionary tree structure."""
        tree = {}

        def build_tree(parent_name: Optional[str] = None) -> Dict[str, Any]:
            children = {}
            for name, cls in self.taxonomy.items():
                if cls.parent == parent_name:
                    children[name] = {
                        "level": cls.level,
                        "description": cls.description,
                        "children": build_tree(name),
                    }
            return children

        return build_tree(None)
