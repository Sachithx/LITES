"""Tests for hierarchy building module."""

import numpy as np
import pytest
from lites.detection import Event
from lites.classification import ClassifiedEvent
from lites.hierarchy import HierarchyBuilder, EventNode


class TestEventNode:
    """Test EventNode."""

    def test_node_creation(self):
        """Test creating an event node."""
        node = EventNode(node_id="test_node")
        assert node.node_id == "test_node"
        assert node.parent_id is None
        assert len(node.children) == 0

    def test_add_child(self):
        """Test adding child nodes."""
        parent = EventNode(node_id="parent")
        child = EventNode(node_id="child")
        parent.add_child(child)

        assert len(parent.children) == 1
        assert child.parent_id == "parent"

    def test_get_depth(self):
        """Test getting node depth."""
        root = EventNode(node_id="root")
        child1 = EventNode(node_id="child1")
        child2 = EventNode(node_id="child2")
        grandchild = EventNode(node_id="grandchild")

        root.add_child(child1)
        child1.add_child(grandchild)
        root.add_child(child2)

        assert root.get_depth() == 2  # root -> child1 -> grandchild

    def test_to_dict(self):
        """Test converting node to dictionary."""
        node = EventNode(node_id="test")
        node_dict = node.to_dict()

        assert node_dict["node_id"] == "test"
        # children only included if present
        node.add_child(EventNode(node_id="child"))
        node_dict = node.to_dict()
        assert "children" in node_dict


class TestHierarchyBuilder:
    """Test HierarchyBuilder."""

    def create_test_events(self):
        """Create test classified events."""
        events = [
            Event(10.0, 1.0, 5.0, "peak"),
            Event(15.0, 1.0, 3.0, "peak"),
            Event(25.0, 5.0, 4.0, "threshold_exceeded"),
        ]
        return [
            ClassifiedEvent(e, ["event", "transient", "spike"]) for e in events
        ]

    def test_temporal_hierarchy(self):
        """Test building temporal hierarchy."""
        builder = HierarchyBuilder(strategy="temporal")
        classified_events = self.create_test_events()
        root = builder.build_hierarchy(classified_events, time_window=10.0)

        assert root is not None
        assert isinstance(root, EventNode)
        assert len(root.children) > 0

    def test_class_based_hierarchy(self):
        """Test building class-based hierarchy."""
        builder = HierarchyBuilder(strategy="class_based")
        classified_events = self.create_test_events()
        root = builder.build_hierarchy(classified_events)

        assert root is not None
        assert isinstance(root, EventNode)

    def test_causal_hierarchy(self):
        """Test building causal hierarchy."""
        builder = HierarchyBuilder(strategy="causal")
        classified_events = self.create_test_events()
        root = builder.build_hierarchy(classified_events, time_threshold=10.0)

        assert root is not None
        assert isinstance(root, EventNode)

    def test_visualize_hierarchy(self):
        """Test visualizing hierarchy as text."""
        builder = HierarchyBuilder(strategy="temporal")
        classified_events = self.create_test_events()
        root = builder.build_hierarchy(classified_events, time_window=10.0)

        visualization = builder.visualize_hierarchy(root)
        assert isinstance(visualization, str)
        assert len(visualization) > 0

    def test_to_dict(self):
        """Test converting hierarchy to dictionary."""
        builder = HierarchyBuilder(strategy="temporal")
        classified_events = self.create_test_events()
        root = builder.build_hierarchy(classified_events, time_window=10.0)

        hierarchy_dict = builder.to_dict(root)
        assert isinstance(hierarchy_dict, dict)
        assert "strategy" in hierarchy_dict
        assert "root" in hierarchy_dict
        assert "total_nodes" in hierarchy_dict

    def test_empty_events(self):
        """Test building hierarchy with no events."""
        builder = HierarchyBuilder(strategy="temporal")
        root = builder.build_hierarchy([])

        assert root is not None
        assert len(root.children) == 0

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            builder = HierarchyBuilder(strategy="invalid")
            builder.build_hierarchy([])
