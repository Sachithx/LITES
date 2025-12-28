"""
LLM formatting module.

Converts hierarchical event structures into formats suitable for language model training.
"""

from typing import List, Dict, Any, Optional
import json
from ..hierarchy.builder import EventNode
from ..classification.classifier import ClassifiedEvent


class LLMFormatter:
    """
    Formats event hierarchies for language model training.

    Supports multiple output formats:
    - JSON: Structured data format
    - Text: Natural language descriptions
    - Conversation: Dialog-style format for chat models
    - Instruction: Instruction-following format
    """

    def __init__(self, format_type: str = "json"):
        """
        Initialize LLM formatter.

        Args:
            format_type: Output format ('json', 'text', 'conversation', 'instruction')
        """
        self.format_type = format_type

    def format_hierarchy(self, root: EventNode, **kwargs) -> Any:
        """
        Format event hierarchy for LLM training.

        Args:
            root: Root node of the event hierarchy
            **kwargs: Format-specific parameters

        Returns:
            Formatted data in the specified format
        """
        if self.format_type == "json":
            return self._format_json(root, **kwargs)
        elif self.format_type == "text":
            return self._format_text(root, **kwargs)
        elif self.format_type == "conversation":
            return self._format_conversation(root, **kwargs)
        elif self.format_type == "instruction":
            return self._format_instruction(root, **kwargs)
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")

    def _format_json(self, root: EventNode, **kwargs) -> Dict[str, Any]:
        """Format as structured JSON."""
        include_metadata = kwargs.get("include_metadata", True)

        def node_to_json(node: EventNode) -> Dict[str, Any]:
            result = {"node_id": node.node_id}

            if node.event:
                event_data = node.event.to_dict()
                if include_metadata:
                    result["event"] = event_data
                else:
                    # Simplified version without metadata
                    result["event"] = {
                        "timestamp": event_data["event"]["timestamp"],
                        "type": event_data["event"]["event_type"],
                        "classes": event_data["classes"],
                    }

            if node.children:
                result["children"] = [node_to_json(child) for child in node.children]

            return result

        return node_to_json(root)

    def _format_text(self, root: EventNode, **kwargs) -> str:
        """Format as natural language text."""
        lines = []
        context = kwargs.get("context", "")

        if context:
            lines.append(f"Context: {context}\n")

        lines.append("Event Hierarchy Analysis:\n")

        def node_to_text(node: EventNode, depth: int = 0) -> None:
            indent = "  " * depth

            if node.event:
                event = node.event.event
                classes = node.event.get_class_path()

                description = (
                    f"{indent}- At timestamp {event.timestamp:.2f}, "
                    f"a {event.event_type} event was detected "
                    f"with magnitude {event.magnitude:.2f} "
                    f"and duration {event.duration}. "
                    f"This event is classified as: {classes}."
                )
                lines.append(description)
            elif node.node_id and node.node_id != "root":
                lines.append(f"{indent}- Group: {node.node_id}")

            for child in node.children:
                node_to_text(child, depth + 1)

        node_to_text(root)
        return "\n".join(lines)

    def _format_conversation(self, root: EventNode, **kwargs) -> List[Dict[str, str]]:
        """Format as conversation for chat models."""
        system_prompt = kwargs.get(
            "system_prompt",
            "You are an expert in time series analysis and event detection. "
            "Analyze the following event hierarchy and provide insights.",
        )

        conversation = [{"role": "system", "content": system_prompt}]

        # Create user message with event data
        event_summary = self._create_event_summary(root)
        user_message = f"Here is the detected event hierarchy:\n\n{event_summary}\n\n"

        query = kwargs.get("query", "What patterns do you observe in these events?")
        user_message += query

        conversation.append({"role": "user", "content": user_message})

        # Optionally add assistant response template
        if kwargs.get("include_response_template", False):
            response = self._generate_response_template(root)
            conversation.append({"role": "assistant", "content": response})

        return conversation

    def _format_instruction(self, root: EventNode, **kwargs) -> Dict[str, str]:
        """Format as instruction-following dataset entry."""
        instruction = kwargs.get(
            "instruction",
            "Analyze the following time series event hierarchy and describe the key patterns.",
        )

        input_data = self._create_event_summary(root)

        output = kwargs.get("output", None)
        if output is None:
            output = self._generate_default_output(root)

        return {"instruction": instruction, "input": input_data, "output": output}

    def _create_event_summary(self, root: EventNode) -> str:
        """Create a summary of events in the hierarchy."""
        events = []

        def collect_events(node: EventNode) -> None:
            if node.event:
                events.append(node.event)
            for child in node.children:
                collect_events(child)

        collect_events(root)

        if not events:
            return "No events detected."

        summary_lines = [f"Total events: {len(events)}\n"]

        # Group by class
        class_counts: Dict[str, int] = {}
        for event in events:
            class_path = event.get_class_path()
            class_counts[class_path] = class_counts.get(class_path, 0) + 1

        summary_lines.append("Event distribution by class:")
        for class_path, count in sorted(class_counts.items()):
            summary_lines.append(f"  - {class_path}: {count}")

        # Time range
        timestamps = [e.event.timestamp for e in events]
        summary_lines.append(
            f"\nTime range: {min(timestamps):.2f} to {max(timestamps):.2f}"
        )

        return "\n".join(summary_lines)

    def _generate_response_template(self, root: EventNode) -> str:
        """Generate a template response for conversation format."""
        events = []

        def collect_events(node: EventNode) -> None:
            if node.event:
                events.append(node.event)
            for child in node.children:
                collect_events(child)

        collect_events(root)

        if not events:
            return "No significant events were detected in this time series."

        response = "Based on the event hierarchy analysis, I observe the following patterns:\n\n"

        # Count event types
        type_counts: Dict[str, int] = {}
        for event in events:
            class_path = event.get_class_path()
            type_counts[class_path] = type_counts.get(class_path, 0) + 1

        response += f"1. Event Distribution: The system detected {len(events)} events total. "

        # Most common type
        if type_counts:
            most_common = max(type_counts.items(), key=lambda x: x[1])
            response += f"The most common event type is '{most_common[0]}' with {most_common[1]} occurrences.\n\n"

        response += (
            "2. Temporal Patterns: Events are organized hierarchically, "
            "showing relationships between different event types and their temporal sequence.\n\n"
        )

        response += (
            "3. Recommendations: Further analysis could explore causal relationships "
            "between events and investigate if there are predictable patterns."
        )

        return response

    def _generate_default_output(self, root: EventNode) -> str:
        """Generate default output for instruction format."""
        return self._generate_response_template(root)

    def format_batch(
        self, hierarchies: List[EventNode], **kwargs
    ) -> List[Any]:
        """
        Format multiple hierarchies as a batch.

        Args:
            hierarchies: List of EventNode root nodes
            **kwargs: Format-specific parameters

        Returns:
            List of formatted outputs
        """
        return [self.format_hierarchy(root, **kwargs) for root in hierarchies]

    def save_to_file(self, formatted_data: Any, filepath: str) -> None:
        """
        Save formatted data to file.

        Args:
            formatted_data: Data to save
            filepath: Output file path
        """
        with open(filepath, "w") as f:
            if isinstance(formatted_data, (dict, list)):
                json.dump(formatted_data, f, indent=2)
            else:
                f.write(str(formatted_data))
