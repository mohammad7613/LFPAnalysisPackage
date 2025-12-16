from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VisualizationSpec:
    """Normalized representation of a visualization entry from the config."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    children: List["VisualizationSpec"] = field(default_factory=list)

    @property
    def is_composite(self) -> bool:
        return bool(self.children)


def build_visualization_spec(spec: Dict[str, Any]) -> VisualizationSpec:
    """Build a VisualizationSpec (recursively) from a raw config dictionary."""

    children = [build_visualization_spec(child) for child in spec.get("children", [])]

    return VisualizationSpec(
        name=spec["name"],
        args=spec.get("args", {}),
        inputs=spec.get("inputs", []),
        children=children,
    )
