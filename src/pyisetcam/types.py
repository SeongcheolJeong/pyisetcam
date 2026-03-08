"""Public dataclass-backed ISET object types."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseISETObject:
    """Common mutable storage for MATLAB-like extensibility."""

    name: str
    type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "BaseISETObject":
        return copy.deepcopy(self)


@dataclass
class Scene(BaseISETObject):
    name: str = "scene"
    type: str = "scene"


@dataclass
class OpticalImage(BaseISETObject):
    name: str = "opticalimage"
    type: str = "opticalimage"


@dataclass
class Sensor(BaseISETObject):
    name: str = "sensor"
    type: str = "sensor"


@dataclass
class ImageProcessor(BaseISETObject):
    name: str = "vcimage"
    type: str = "vcimage"


@dataclass
class Display(BaseISETObject):
    name: str = "display"
    type: str = "display"


@dataclass
class Camera(BaseISETObject):
    name: str = "camera"
    type: str = "camera"


@dataclass
class SessionContext:
    """Optional vcSESSION-style object registry for compatibility workflows."""

    name: str = "vcSESSION"
    objects: dict[str, dict[int, BaseISETObject]] = field(default_factory=dict)
    selected: dict[str, int | None] = field(default_factory=dict)
    next_ids: dict[str, int] = field(default_factory=dict)
