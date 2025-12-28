"""Tests for LLM formatting module."""

import numpy as np
import pytest
import json
from lites.detection import Event
from lites.classification import ClassifiedEvent
from lites.hierarchy import EventNode
from lites.formatting import LLMFormatter


class TestLLMFormatter:
    """Test LLMFormatter."""

    def create_test_hierarchy(self):
        """Create a simple test hierarchy."""
        root = EventNode(node_id="root")

        event1 = Event(10.0, 1.0, 5.0, "peak")
        classified1 = ClassifiedEvent(event1, ["event", "transient", "spike"])
        node1 = EventNode(event=classified1)

        event2 = Event(20.0, 5.0, 3.0, "threshold_exceeded")
        classified2 = ClassifiedEvent(event2, ["event", "sustained", "step_up"])
        node2 = EventNode(event=classified2)

        root.add_child(node1)
        root.add_child(node2)

        return root

    def test_json_format(self):
        """Test JSON formatting."""
        formatter = LLMFormatter(format_type="json")
        root = self.create_test_hierarchy()
        output = formatter.format_hierarchy(root)

        assert isinstance(output, dict)
        assert "node_id" in output
        assert "children" in output

    def test_text_format(self):
        """Test text formatting."""
        formatter = LLMFormatter(format_type="text")
        root = self.create_test_hierarchy()
        output = formatter.format_hierarchy(root)

        assert isinstance(output, str)
        assert "Event Hierarchy" in output
        assert len(output) > 0

    def test_conversation_format(self):
        """Test conversation formatting."""
        formatter = LLMFormatter(format_type="conversation")
        root = self.create_test_hierarchy()
        output = formatter.format_hierarchy(root)

        assert isinstance(output, list)
        assert len(output) >= 2  # At least system and user messages
        assert all("role" in msg and "content" in msg for msg in output)

    def test_instruction_format(self):
        """Test instruction formatting."""
        formatter = LLMFormatter(format_type="instruction")
        root = self.create_test_hierarchy()
        output = formatter.format_hierarchy(root)

        assert isinstance(output, dict)
        assert "instruction" in output
        assert "input" in output
        assert "output" in output

    def test_format_batch(self):
        """Test batch formatting."""
        formatter = LLMFormatter(format_type="json")
        hierarchies = [self.create_test_hierarchy() for _ in range(3)]
        outputs = formatter.format_batch(hierarchies)

        assert len(outputs) == 3
        assert all(isinstance(o, dict) for o in outputs)

    def test_empty_hierarchy(self):
        """Test formatting empty hierarchy."""
        formatter = LLMFormatter(format_type="json")
        root = EventNode(node_id="root")
        output = formatter.format_hierarchy(root)

        assert isinstance(output, dict)

    def test_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError):
            formatter = LLMFormatter(format_type="invalid")
            formatter.format_hierarchy(EventNode())

    def test_save_to_file(self, tmp_path):
        """Test saving formatted data to file."""
        formatter = LLMFormatter(format_type="json")
        root = self.create_test_hierarchy()
        output = formatter.format_hierarchy(root)

        filepath = tmp_path / "output.json"
        formatter.save_to_file(output, str(filepath))

        assert filepath.exists()

        # Verify content
        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert loaded == output
