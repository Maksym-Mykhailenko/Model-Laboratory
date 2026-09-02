"""Explicit assumptions, ambiguities and numerical diagnostics for Model Laboratory.

Assumptions and ambiguities are part of the formal model record rather than informal UI
text.  Numerical diagnostics describe the quality of a deterministic computation without
altering the supplied model or silently converting a failed/partial computation into a
successful one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .provenance import Provenance


class AmbiguityStatus(str, Enum):
    """Resolution state of a declared ambiguity."""

    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class Assumption:
    """One explicit assumption associated with the supplied model."""

    name: str
    statement: str
    affects: tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))


@dataclass(frozen=True, slots=True)
class Ambiguity:
    """One explicit ambiguity that may or may not block computation.

    An empty ``affects`` tuple means the ambiguity concerns the model globally.  A
    blocking unresolved ambiguity prevents deterministic analyses and model renderings;
    the model can still be parsed and inspected.
    """

    name: str
    statement: str
    options: tuple[str, ...] = ()
    resolution: str | None = None
    blocking: bool = True
    affects: tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=lambda: Provenance.supplied("unspecified"))

    @property
    def status(self) -> AmbiguityStatus:
        return (
            AmbiguityStatus.RESOLVED
            if self.resolution is not None
            else AmbiguityStatus.UNRESOLVED
        )

    @property
    def is_resolved(self) -> bool:
        return self.status is AmbiguityStatus.RESOLVED

    @property
    def blocks_computation(self) -> bool:
        return self.blocking and not self.is_resolved


class NumericalStatus(str, Enum):
    """Overall status of a numerical computation."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    INDETERMINATE = "indeterminate"
    FAILED = "failed"


class DiagnosticSeverity(str, Enum):
    """Severity of one numerical diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class NumericalDiagnostic:
    """Machine-readable diagnostic emitted by a numerical computation.

    Scientific identity consists of ``code``, ``severity`` and canonical structured
    ``details``.  ``message`` is human-facing presentation text: it is preserved in
    experiment files and reports but does not determine numerical reproduction.
    """

    code: str
    severity: DiagnosticSeverity
    message: str
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        canonical = tuple(sorted(((str(key), str(value)) for key, value in self.details)))
        object.__setattr__(self, "details", canonical)

    @property
    def scientific_identity(self) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        """Return the canonical diagnostic identity used for numerical comparison."""
        return (self.code, self.severity.value, self.details)

    @classmethod
    def info(cls, code: str, message: str, **details: object) -> "NumericalDiagnostic":
        return cls(
            code=code,
            severity=DiagnosticSeverity.INFO,
            message=message,
            details=tuple((key, str(value)) for key, value in details.items()),
        )

    @classmethod
    def warning(
        cls, code: str, message: str, **details: object
    ) -> "NumericalDiagnostic":
        return cls(
            code=code,
            severity=DiagnosticSeverity.WARNING,
            message=message,
            details=tuple((key, str(value)) for key, value in details.items()),
        )

    @classmethod
    def error(cls, code: str, message: str, **details: object) -> "NumericalDiagnostic":
        return cls(
            code=code,
            severity=DiagnosticSeverity.ERROR,
            message=message,
            details=tuple((key, str(value)) for key, value in details.items()),
        )


@dataclass(frozen=True, slots=True)
class NumericalStatistics:
    """Small generic counter collection attached to a numerical result."""

    values: tuple[tuple[str, int | float], ...] = ()

    def as_dict(self) -> dict[str, int | float]:
        return dict(self.values)
