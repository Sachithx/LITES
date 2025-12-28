"""
Hierarchy building module.

Organizes classified events into hierarchical structures with parent-child relationships.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from ..classification.classifier import ClassifiedEvent


class EventNode:
    """Represents a node in the event hierarchy tree."""

    def __init__(
        self,
        event: Optional[ClassifiedEvent] = None,
        node_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        children: Optional[List["EventNode"]] = None,
    ):
        self.event = event
        self.node_id = node_id or self._generate_id()
        self.parent_id = parent_id
        self.children = children or []

    def _generate_id(self) -> str:
        """Generate a unique node ID."""
        if self.event:
            return f"node_{self.event.event.timestamp}_{id(self.event)}"
        return f"node_{id(self)}"

    def add_child(self, child: "EventNode") -> None:
        """Add a child node."""
        child.parent_id = self.node_id
        self.children.append(child)

    def to_dict(self, include_children: bool = True) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
        }

        if self.event:
            result["event"] = self.event.to_dict()

        if include_children and self.children:
            result["children"] = [child.to_dict(include_children=True) for child in self.children]

        return result

    def get_depth(self) -> int:
        """Get the depth of this node in the tree."""
        if not self.children:
            return 0
        return 1 + max(child.get_depth() for child in self.children)

    def __repr__(self) -> str:
        event_info = f", event={self.event}" if self.event else ""
        return f"EventNode(id={self.node_id}, children={len(self.children)}{event_info})"


class HierarchyBuilder:
    """
    Builds hierarchical structures from classified events.

    Supports multiple organization strategies:
    - Temporal: Organize by time windows
    - Class-based: Organize by event classes
    - Causal: Organize by potential causal relationships
    - Spatial: Organize by spatial proximity (if location data available)
    """

    def __init__(self, strategy: str = "temporal"):
        """
        Initialize hierarchy builder.

        Args:
            strategy: Organization strategy ('temporal', 'class_based', 'causal')
        """
        self.strategy = strategy

    def build_hierarchy(
        self, classified_events: List[ClassifiedEvent], **kwargs
    ) -> EventNode:
        """
        Build event hierarchy from classified events.

        Args:
            classified_events: List of ClassifiedEvent objects
            **kwargs: Strategy-specific parameters

        Returns:
            Root EventNode of the hierarchy tree
        """
        if self.strategy == "temporal":
            return self._build_temporal_hierarchy(classified_events, **kwargs)
        elif self.strategy == "class_based":
            return self._build_class_hierarchy(classified_events, **kwargs)
        elif self.strategy == "causal":
            return self._build_causal_hierarchy(classified_events, **kwargs)
        else:
            raise ValueError(f"Unknown hierarchy strategy: {self.strategy}")

    def _build_temporal_hierarchy(
        self, classified_events: List[ClassifiedEvent], **kwargs
    ) -> EventNode:
        """Build hierarchy based on temporal relationships."""
        time_window = kwargs.get("time_window", 10.0)

        # Sort events by timestamp
        sorted_events = sorted(classified_events, key=lambda e: e.event.timestamp)

        # Create root node
        root = EventNode(node_id="root_temporal")

        if not sorted_events:
            return root

        # Group events into time windows
        current_window_start = sorted_events[0].event.timestamp
        current_window_events = []

        for classified_event in sorted_events:
            event_time = classified_event.event.timestamp

            if event_time - current_window_start <= time_window:
                current_window_events.append(classified_event)
            else:
                # Create a window node
                window_node = EventNode(
                    node_id=f"window_{current_window_start}_{current_window_start + time_window}"
                )

                # Add events to window
                for event in current_window_events:
                    event_node = EventNode(event=event)
                    window_node.add_child(event_node)

                root.add_child(window_node)

                # Start new window
                current_window_start = event_time
                current_window_events = [classified_event]

        # Add final window
        if current_window_events:
            window_node = EventNode(
                node_id=f"window_{current_window_start}_{current_window_start + time_window}"
            )
            for event in current_window_events:
                event_node = EventNode(event=event)
                window_node.add_child(event_node)
            root.add_child(window_node)

        return root

    def _build_class_hierarchy(
        self, classified_events: List[ClassifiedEvent], **kwargs
    ) -> EventNode:
        """Build hierarchy based on event classes."""
        # Create root node
        root = EventNode(node_id="root_class_based")

        # Group events by their class paths
        class_groups: Dict[str, List[ClassifiedEvent]] = {}
        for classified_event in classified_events:
            class_path = classified_event.get_class_path()
            if class_path not in class_groups:
                class_groups[class_path] = []
            class_groups[class_path].append(classified_event)

        # Create hierarchy based on class structure
        for class_path, events in class_groups.items():
            # Create intermediate nodes for each level in the class hierarchy
            # Use classes from the first event in this group (all have same class path)
            classes = events[0].classes
            current_parent = root

            for i, class_name in enumerate(classes[:-1]):  # Exclude the leaf class
                node_id = f"class_{'>'.join(classes[:i+1])}"

                # Check if node already exists
                existing_node = None
                for child in current_parent.children:
                    if child.node_id == node_id:
                        existing_node = child
                        break

                if existing_node:
                    current_parent = existing_node
                else:
                    new_node = EventNode(node_id=node_id)
                    current_parent.add_child(new_node)
                    current_parent = new_node

            # Add leaf events
            for event in events:
                event_node = EventNode(event=event)
                current_parent.add_child(event_node)

        return root

    def _build_causal_hierarchy(
        self, classified_events: List[ClassifiedEvent], **kwargs
    ) -> EventNode:
        """Build hierarchy based on potential causal relationships."""
        time_threshold = kwargs.get("time_threshold", 5.0)

        # Sort events by timestamp
        sorted_events = sorted(classified_events, key=lambda e: e.event.timestamp)

        # Create root node
        root = EventNode(node_id="root_causal")

        if not sorted_events:
            return root

        # Track potential parent events
        potential_parents = []

        for classified_event in sorted_events:
            event_time = classified_event.event.timestamp
            event_node = EventNode(event=classified_event)

            # Find potential parent (recent event that could have caused this one)
            parent_found = False
            for parent_event, parent_node in reversed(potential_parents):
                time_diff = event_time - parent_event.event.timestamp

                if time_diff <= time_threshold:
                    # Check if there's a potential causal relationship
                    if self._is_potential_cause(parent_event, classified_event):
                        parent_node.add_child(event_node)
                        parent_found = True
                        break

            # If no parent found, add to root
            if not parent_found:
                root.add_child(event_node)

            # Add to potential parents
            potential_parents.append((classified_event, event_node))

            # Remove old potential parents
            potential_parents = [
                (e, n)
                for e, n in potential_parents
                if event_time - e.event.timestamp <= time_threshold
            ]

        return root

    def _is_potential_cause(
        self, parent_event: ClassifiedEvent, child_event: ClassifiedEvent
    ) -> bool:
        """Determine if parent event could have caused child event."""
        # Simple heuristic: parent should occur before child
        if parent_event.event.timestamp >= child_event.event.timestamp:
            return False

        # Check if events are of related types
        # For example, a spike might cause an anomaly
        parent_type = parent_event.event.event_type
        child_type = child_event.event.event_type

        # Simple causal rules
        if parent_type == "changepoint" and child_type in ["anomaly", "threshold_exceeded"]:
            return True
        if parent_type == "peak" and child_type == "anomaly":
            return True

        return False

    def visualize_hierarchy(self, root: EventNode, max_depth: int = 10) -> str:
        """
        Generate a text-based visualization of the hierarchy.

        Args:
            root: Root node of the hierarchy
            max_depth: Maximum depth to visualize

        Returns:
            String representation of the hierarchy tree
        """
        lines = []

        def traverse(node: EventNode, depth: int = 0, prefix: str = "") -> None:
            if depth > max_depth:
                return

            indent = "  " * depth
            if node.event:
                lines.append(f"{indent}{prefix}└─ {node.event}")
            else:
                lines.append(f"{indent}{prefix}└─ Node({node.node_id})")

            for i, child in enumerate(node.children):
                is_last = i == len(node.children) - 1
                child_prefix = "  " if is_last else "│ "
                traverse(child, depth + 1, child_prefix)

        traverse(root)
        return "\n".join(lines)

    def to_dict(self, root: EventNode) -> Dict[str, Any]:
        """Convert hierarchy to dictionary representation."""
        return {
            "strategy": self.strategy,
            "root": root.to_dict(include_children=True),
            "total_nodes": self._count_nodes(root),
            "max_depth": root.get_depth(),
        }

    def _count_nodes(self, node: EventNode) -> int:
        """Count total nodes in hierarchy."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
