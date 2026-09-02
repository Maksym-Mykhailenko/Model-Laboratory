"""Public capability identifiers and reports for Model Laboratory.

Capability discovery is implemented by :mod:`model_lab.registry`.  This module keeps the
stable public identifiers separate from the registry implementation so analysis,
visualisation and interface code can refer to capabilities without importing the registry
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .model import ModelIR


class ModelVisualisation(str, Enum):
    """Direct visualisations of a supplied scalar function."""

    TWO_D_FUNCTION_PLOT = "2D function plot"
    THREE_D_SURFACE = "3D surface"
    CONTOUR_MAP = "contour map"
    HEAT_MAP = "heat map"


class AnalysisCapability(str, Enum):
    """Deterministic mathematical analyses supported by the current core."""

    FIRST_DERIVATIVE = "first derivative"
    SECOND_DERIVATIVE = "second derivative"
    GRADIENT = "gradient"
    HESSIAN = "Hessian"
    STATIONARY_POINTS = "stationary-point analysis"
    PARAMETER_SWEEP = "parameter sweep"


class AnalysisVisualisation(str, Enum):
    """Visualisations of analysis results rather than direct model evaluations."""

    STATIONARY_POINT_OVERLAY = "stationary-point overlay"
    SWEEP_CLASSIFICATION_COUNTS = "stationary-point counts"
    SWEEP_POSITIONS = "stationary-point positions"
    SWEEP_FUNCTION_VALUES = "stationary function values"


class ControlCapability(str, Enum):
    """Interactive controls generated directly from the model structure."""

    PARAMETER_CONTROLS = "parameter controls"


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """All capabilities currently applicable to one validated Model IR."""

    visualisations: tuple[ModelVisualisation, ...]
    analyses: tuple[AnalysisCapability, ...]
    controls: tuple[ControlCapability, ...]


def detect_capabilities(model: ModelIR) -> CapabilityReport:
    """Return all currently registered capabilities applicable to ``model``.

    The import is deliberately local: ``registry`` imports these public identifiers while
    constructing the default registry, so keeping this dependency lazy avoids a circular
    import and preserves a small public API.
    """
    from .registry import capability_registry

    return capability_registry.detect(model)


def compatible_analysis_visualisations(
    analysis: AnalysisCapability,
    model: ModelIR,
) -> tuple[AnalysisVisualisation, ...]:
    """Return registered visualisations compatible with ``analysis`` for ``model``."""
    from .registry import capability_registry

    return capability_registry.compatible_analysis_visualisations(analysis, model)
