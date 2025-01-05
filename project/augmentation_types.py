from dataclasses import dataclass
from enum import Enum, auto

class AugmentationType(Enum):
    SHIFT_X = auto()
    SHIFT_Y = auto()
    ROTATION = auto()
    BRIGHTNESS = auto()
    CONTRAST = auto()
    BLUR = auto()
    NOISE = auto()
    ZOOM = auto()

    def __str__(self):
        return self.name.replace('_', ' ').title()

@dataclass
class AugmentationConfig:
    type: AugmentationType
    min_value: float  # Percentage
    max_value: float  # Percentage