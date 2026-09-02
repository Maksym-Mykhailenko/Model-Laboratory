"""Declarative capability registry for Model Laboratory.

The registry is the single source of truth for what the current laboratory can do.  Each
capability declares the structural requirements that make it applicable.  Analysis
capabilities additionally declare their deterministic runner, result type and compatible
analysis visualisations.  Analysis visualisations declare which result types they accept.

No capability implementation changes the supplied model.  The registry only inspects a
validated :class:`~model_lab.model.ModelIR` and describes or dispatches existing
capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .analysis import (
    ParameterSweepResult,
    StationaryPointAnalysisResult,
    run_parameter_sweep,
    run_stationary_point_analysis,
)
from .capabilities import (
    AnalysisCapability,
    AnalysisVisualisation,
    CapabilityReport,
    ControlCapability,
    ModelVisualisation,
)
from .evaluator import FunctionEvaluation, SurfaceEvaluation
from .model import ModelIR
from .symbolic import (
    OneVariableSymbolicAnalysis,
    TwoVariableSymbolicAnalysis,
    analyse_one_variable_function,
    analyse_two_variable_function,
)


CapabilityKey = ModelVisualisation | AnalysisCapability | AnalysisVisualisation | ControlCapability
AnalysisRunner = Callable[..., object]
RequirementPredicate = Callable[[ModelIR], bool]


class CapabilityKind(str, Enum):
    """Registry category for a capability definition."""

    MODEL_VISUALISATION = "model visualisation"
    ANALYSIS = "analysis"
    ANALYSIS_VISUALISATION = "analysis visualisation"
    CONTROL = "control"


class AnalysisVisualisationMode(str, Enum):
    """How an analysis visualisation is composed with the rest of the interface."""

    OVERLAY = "overlay"
    STANDALONE = "standalone"


@dataclass(frozen=True, slots=True)
class ModelRequirement:
    """One explicit structural condition that a Model IR may satisfy."""

    identifier: str
    description: str
    predicate: RequirementPredicate

    def is_satisfied_by(self, model: ModelIR) -> bool:
        return bool(self.predicate(model))


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    """Declarative metadata for one registered capability."""

    key: CapabilityKey
    kind: CapabilityKind
    requirements: tuple[ModelRequirement, ...] = ()
    runner: AnalysisRunner | None = None
    result_type: type | None = None
    compatible_visualisations: tuple[AnalysisVisualisation, ...] = ()
    accepted_result_types: tuple[type, ...] = ()
    visualisation_mode: AnalysisVisualisationMode | None = None

    def is_applicable_to(self, model: ModelIR) -> bool:
        return all(requirement.is_satisfied_by(model) for requirement in self.requirements)

    def unmet_requirements(self, model: ModelIR) -> tuple[ModelRequirement, ...]:
        return tuple(
            requirement
            for requirement in self.requirements
            if not requirement.is_satisfied_by(model)
        )


class CapabilityRegistryError(ValueError):
    """Raised when the capability registry is malformed or misused."""


class CapabilityRegistry:
    """Validated collection of declarative capability definitions."""

    def __init__(self, definitions: tuple[CapabilityDefinition, ...] = ()) -> None:
        self._definitions: dict[CapabilityKey, CapabilityDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: CapabilityDefinition) -> None:
        """Register one definition, rejecting duplicate capability identifiers."""
        if definition.key in self._definitions:
            raise CapabilityRegistryError(
                f"Capability '{definition.key.value}' is already registered."
            )
        self._validate_definition(definition)
        self._definitions[definition.key] = definition

    def _validate_definition(self, definition: CapabilityDefinition) -> None:
        if definition.kind == CapabilityKind.ANALYSIS:
            if not isinstance(definition.key, AnalysisCapability):
                raise CapabilityRegistryError("Analysis definitions require an AnalysisCapability key.")
            if definition.runner is None or definition.result_type is None:
                raise CapabilityRegistryError(
                    f"Analysis '{definition.key.value}' must declare a runner and result type."
                )

        elif definition.kind == CapabilityKind.MODEL_VISUALISATION:
            if not isinstance(definition.key, ModelVisualisation):
                raise CapabilityRegistryError(
                    "Model-visualisation definitions require a ModelVisualisation key."
                )
            if not definition.accepted_result_types:
                raise CapabilityRegistryError(
                    f"Model visualisation '{definition.key.value}' must declare accepted evaluation types."
                )

        elif definition.kind == CapabilityKind.ANALYSIS_VISUALISATION:
            if not isinstance(definition.key, AnalysisVisualisation):
                raise CapabilityRegistryError(
                    "Analysis-visualisation definitions require an AnalysisVisualisation key."
                )
            if not definition.accepted_result_types:
                raise CapabilityRegistryError(
                    f"Analysis visualisation '{definition.key.value}' must declare accepted result types."
                )
            if definition.visualisation_mode is None:
                raise CapabilityRegistryError(
                    f"Analysis visualisation '{definition.key.value}' must declare a composition mode."
                )

        elif definition.kind == CapabilityKind.CONTROL:
            if not isinstance(definition.key, ControlCapability):
                raise CapabilityRegistryError("Control definitions require a ControlCapability key.")

    @property
    def definitions(self) -> tuple[CapabilityDefinition, ...]:
        """Return definitions in deterministic registration order."""
        return tuple(self._definitions.values())

    def definition(self, key: CapabilityKey) -> CapabilityDefinition:
        try:
            return self._definitions[key]
        except KeyError as exc:
            raise CapabilityRegistryError(f"Capability '{key.value}' is not registered.") from exc

    def applicable_definitions(
        self,
        model: ModelIR,
        *,
        kind: CapabilityKind | None = None,
    ) -> tuple[CapabilityDefinition, ...]:
        return tuple(
            definition
            for definition in self._definitions.values()
            if (kind is None or definition.kind == kind) and definition.is_applicable_to(model)
        )

    def detect(self, model: ModelIR) -> CapabilityReport:
        """Construct a public capability report entirely from registry definitions."""
        visualisations = tuple(
            definition.key
            for definition in self.applicable_definitions(
                model, kind=CapabilityKind.MODEL_VISUALISATION
            )
        )
        analyses = tuple(
            definition.key
            for definition in self.applicable_definitions(model, kind=CapabilityKind.ANALYSIS)
        )
        controls = tuple(
            definition.key
            for definition in self.applicable_definitions(model, kind=CapabilityKind.CONTROL)
        )
        return CapabilityReport(
            visualisations=visualisations,  # type: ignore[arg-type]
            analyses=analyses,  # type: ignore[arg-type]
            controls=controls,  # type: ignore[arg-type]
        )

    def compatible_analysis_visualisations(
        self,
        analysis: AnalysisCapability,
        model: ModelIR,
    ) -> tuple[AnalysisVisualisation, ...]:
        """Return renderers whose contracts match an applicable analysis result."""
        analysis_definition = self.definition(analysis)
        if analysis_definition.kind != CapabilityKind.ANALYSIS:
            raise CapabilityRegistryError(f"'{analysis.value}' is not registered as an analysis.")
        if not analysis_definition.is_applicable_to(model):
            return ()

        result_type = analysis_definition.result_type
        compatible: list[AnalysisVisualisation] = []
        for visualisation in analysis_definition.compatible_visualisations:
            visualisation_definition = self.definition(visualisation)
            if visualisation_definition.kind != CapabilityKind.ANALYSIS_VISUALISATION:
                raise CapabilityRegistryError(
                    f"'{visualisation.value}' is not registered as an analysis visualisation."
                )
            if result_type is not None and any(
                issubclass(result_type, accepted_type)
                for accepted_type in visualisation_definition.accepted_result_types
            ):
                compatible.append(visualisation)
        return tuple(compatible)

    def run_analysis(
        self,
        analysis: AnalysisCapability,
        model: ModelIR,
        /,
        **kwargs: Any,
    ) -> object:
        """Execute a registered deterministic analysis after checking applicability."""
        definition = self.definition(analysis)
        if definition.kind != CapabilityKind.ANALYSIS or definition.runner is None:
            raise CapabilityRegistryError(f"'{analysis.value}' is not an executable analysis.")
        if not definition.is_applicable_to(model):
            unmet = "; ".join(
                requirement.description for requirement in definition.unmet_requirements(model)
            )
            raise CapabilityRegistryError(
                f"Analysis '{analysis.value}' is not applicable to this model"
                + (f": {unmet}." if unmet else ".")
            )

        result = definition.runner(model, **kwargs)
        if definition.result_type is not None and not isinstance(result, definition.result_type):
            raise CapabilityRegistryError(
                f"Analysis '{analysis.value}' returned {type(result).__name__}; "
                f"expected {definition.result_type.__name__}."
            )
        return result


# Reusable structural requirements.  These are intentionally stated in terms of the
# current IR rather than discipline-specific semantics.
EXACTLY_ONE_VARIABLE = ModelRequirement(
    "exactly_one_variable",
    "exactly one continuous variable is required",
    lambda model: len(model.variables) == 1,
)
EXACTLY_TWO_VARIABLES = ModelRequirement(
    "exactly_two_variables",
    "exactly two continuous variables are required",
    lambda model: len(model.variables) == 2,
)
ONE_OR_TWO_VARIABLES = ModelRequirement(
    "one_or_two_variables",
    "one or two continuous variables are required",
    lambda model: len(model.variables) in (1, 2),
)
EXACTLY_ONE_FUNCTION = ModelRequirement(
    "exactly_one_function",
    "exactly one scalar function is required",
    lambda model: len(model.functions) == 1,
)
HAS_PARAMETERS = ModelRequirement(
    "has_parameters",
    "at least one adjustable parameter is required",
    lambda model: bool(model.parameters),
)


NO_BLOCKING_AMBIGUITIES = ModelRequirement(
    "no_blocking_ambiguities",
    "all blocking ambiguities must be resolved",
    lambda model: model.is_computation_ready,
)

NO_CONSTRAINTS = ModelRequirement(
    "no_constraints",
    "stationary-point analysis currently requires a model without declared constraints",
    lambda model: not model.constraints,
)


capability_registry = CapabilityRegistry(
    definitions=(
        # Direct model visualisations.
        CapabilityDefinition(
            ModelVisualisation.TWO_D_FUNCTION_PLOT,
            CapabilityKind.MODEL_VISUALISATION,
            requirements=(NO_BLOCKING_AMBIGUITIES, EXACTLY_ONE_VARIABLE, EXACTLY_ONE_FUNCTION),
            accepted_result_types=(FunctionEvaluation,),
        ),
        CapabilityDefinition(
            ModelVisualisation.THREE_D_SURFACE,
            CapabilityKind.MODEL_VISUALISATION,
            requirements=(NO_BLOCKING_AMBIGUITIES, EXACTLY_TWO_VARIABLES, EXACTLY_ONE_FUNCTION),
            accepted_result_types=(SurfaceEvaluation,),
        ),
        CapabilityDefinition(
            ModelVisualisation.CONTOUR_MAP,
            CapabilityKind.MODEL_VISUALISATION,
            requirements=(NO_BLOCKING_AMBIGUITIES, EXACTLY_TWO_VARIABLES, EXACTLY_ONE_FUNCTION),
            accepted_result_types=(SurfaceEvaluation,),
        ),
        CapabilityDefinition(
            ModelVisualisation.HEAT_MAP,
            CapabilityKind.MODEL_VISUALISATION,
            requirements=(NO_BLOCKING_AMBIGUITIES, EXACTLY_TWO_VARIABLES, EXACTLY_ONE_FUNCTION),
            accepted_result_types=(SurfaceEvaluation,),
        ),

        # Deterministic analyses.
        CapabilityDefinition(
            AnalysisCapability.FIRST_DERIVATIVE,
            CapabilityKind.ANALYSIS,
            requirements=(NO_BLOCKING_AMBIGUITIES, EXACTLY_ONE_VARIABLE, EXACTLY_ONE_FUNCTION),
            runner=analyse_one_variable_function,
            result_type=OneVariableSymbolicAnalysis,
        ),
        CapabilityDefinition(
            AnalysisCapability.SECOND_DERIVATIVE,
            CapabilityKind.ANALYSIS,
            requirements=(NO_BLOCKING_AMBIGUITIES, EXACTLY_ONE_VARIABLE, EXACTLY_ONE_FUNCTION),
            runner=analyse_one_variable_function,
            result_type=OneVariableSymbolicAnalysis,
        ),
        CapabilityDefinition(
            AnalysisCapability.GRADIENT,
            CapabilityKind.ANALYSIS,
            requirements=(NO_BLOCKING_AMBIGUITIES, EXACTLY_TWO_VARIABLES, EXACTLY_ONE_FUNCTION),
            runner=analyse_two_variable_function,
            result_type=TwoVariableSymbolicAnalysis,
        ),
        CapabilityDefinition(
            AnalysisCapability.HESSIAN,
            CapabilityKind.ANALYSIS,
            requirements=(NO_BLOCKING_AMBIGUITIES, EXACTLY_TWO_VARIABLES, EXACTLY_ONE_FUNCTION),
            runner=analyse_two_variable_function,
            result_type=TwoVariableSymbolicAnalysis,
        ),
        CapabilityDefinition(
            AnalysisCapability.STATIONARY_POINTS,
            CapabilityKind.ANALYSIS,
            requirements=(NO_BLOCKING_AMBIGUITIES, ONE_OR_TWO_VARIABLES, EXACTLY_ONE_FUNCTION, NO_CONSTRAINTS),
            runner=run_stationary_point_analysis,
            result_type=StationaryPointAnalysisResult,
            compatible_visualisations=(AnalysisVisualisation.STATIONARY_POINT_OVERLAY,),
        ),
        CapabilityDefinition(
            AnalysisCapability.PARAMETER_SWEEP,
            CapabilityKind.ANALYSIS,
            requirements=(NO_BLOCKING_AMBIGUITIES, ONE_OR_TWO_VARIABLES, EXACTLY_ONE_FUNCTION, HAS_PARAMETERS, NO_CONSTRAINTS),
            runner=run_parameter_sweep,
            result_type=ParameterSweepResult,
            compatible_visualisations=(
                AnalysisVisualisation.SWEEP_CLASSIFICATION_COUNTS,
                AnalysisVisualisation.SWEEP_POSITIONS,
                AnalysisVisualisation.SWEEP_FUNCTION_VALUES,
            ),
        ),

        # Analysis-result visualisations.  These declare result compatibility; actual Plotly
        # rendering remains isolated in visualisation.py.
        CapabilityDefinition(
            AnalysisVisualisation.STATIONARY_POINT_OVERLAY,
            CapabilityKind.ANALYSIS_VISUALISATION,
            accepted_result_types=(StationaryPointAnalysisResult,),
            visualisation_mode=AnalysisVisualisationMode.OVERLAY,
        ),
        CapabilityDefinition(
            AnalysisVisualisation.SWEEP_CLASSIFICATION_COUNTS,
            CapabilityKind.ANALYSIS_VISUALISATION,
            accepted_result_types=(ParameterSweepResult,),
            visualisation_mode=AnalysisVisualisationMode.STANDALONE,
        ),
        CapabilityDefinition(
            AnalysisVisualisation.SWEEP_POSITIONS,
            CapabilityKind.ANALYSIS_VISUALISATION,
            accepted_result_types=(ParameterSweepResult,),
            visualisation_mode=AnalysisVisualisationMode.STANDALONE,
        ),
        CapabilityDefinition(
            AnalysisVisualisation.SWEEP_FUNCTION_VALUES,
            CapabilityKind.ANALYSIS_VISUALISATION,
            accepted_result_types=(ParameterSweepResult,),
            visualisation_mode=AnalysisVisualisationMode.STANDALONE,
        ),

        # Generated controls.
        CapabilityDefinition(
            ControlCapability.PARAMETER_CONTROLS,
            CapabilityKind.CONTROL,
            requirements=(HAS_PARAMETERS,),
        ),
    )
)
