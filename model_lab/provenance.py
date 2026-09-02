"""Explicit provenance primitives for Model Laboratory.

Provenance records where formal content came from without changing its mathematical
meaning.  Supplied, deterministically derived and suggested content remain distinct.
Suggested content also carries an approval state so a proposal can never be confused
with an accepted part of a model or analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProvenanceKind(str, Enum):
    """How an element or result entered the laboratory."""

    SUPPLIED = "supplied"
    DERIVED = "derived"
    SUGGESTED = "suggested"


class ProvenanceApproval(str, Enum):
    """Whether explicit user approval is relevant to an item's provenance."""

    NOT_REQUIRED = "not required"
    PENDING = "pending"
    APPROVED = "approved"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Machine-readable origin and lineage for one model element or analysis result.

    ``source_location`` identifies where directly supplied content appeared in the model
    specification. ``source_refs`` identifies formal model elements or analysis results
    used to derive or suggest the current item. ``operation`` records the deterministic
    transformation or proposed construction that produced it.
    """

    kind: ProvenanceKind
    source_location: str | None = None
    source_refs: tuple[str, ...] = ()
    operation: str | None = None
    approval: ProvenanceApproval = ProvenanceApproval.NOT_REQUIRED
    note: str | None = None

    def __post_init__(self) -> None:
        if self.kind is ProvenanceKind.SUPPLIED and self.approval is not ProvenanceApproval.NOT_REQUIRED:
            raise ValueError("Supplied provenance does not require approval.")
        if self.kind is ProvenanceKind.DERIVED and self.approval is not ProvenanceApproval.NOT_REQUIRED:
            raise ValueError("Deterministically derived provenance does not require approval.")
        if self.kind is ProvenanceKind.SUGGESTED and self.approval is ProvenanceApproval.NOT_REQUIRED:
            raise ValueError("Suggested provenance must be pending or approved.")
        if self.kind is ProvenanceKind.SUPPLIED and not self.source_location:
            raise ValueError("Supplied provenance requires a source location.")
        if self.kind is not ProvenanceKind.SUPPLIED and not self.source_refs:
            raise ValueError("Derived or suggested provenance requires at least one source reference.")
        if self.kind is not ProvenanceKind.SUPPLIED and not self.operation:
            raise ValueError("Derived or suggested provenance requires an operation.")

    @classmethod
    def supplied(cls, source_location: str, *, note: str | None = None) -> "Provenance":
        return cls(
            kind=ProvenanceKind.SUPPLIED,
            source_location=source_location,
            note=note,
        )

    @classmethod
    def derived(
        cls,
        source_refs: tuple[str, ...],
        operation: str,
        *,
        note: str | None = None,
    ) -> "Provenance":
        return cls(
            kind=ProvenanceKind.DERIVED,
            source_refs=source_refs,
            operation=operation,
            note=note,
        )

    @classmethod
    def suggested(
        cls,
        source_refs: tuple[str, ...],
        operation: str,
        *,
        approved: bool = False,
        note: str | None = None,
    ) -> "Provenance":
        return cls(
            kind=ProvenanceKind.SUGGESTED,
            source_refs=source_refs,
            operation=operation,
            approval=(
                ProvenanceApproval.APPROVED if approved else ProvenanceApproval.PENDING
            ),
            note=note,
        )

    @property
    def is_active(self) -> bool:
        """Whether the item may be treated as active rather than merely proposed."""
        return not (
            self.kind is ProvenanceKind.SUGGESTED
            and self.approval is ProvenanceApproval.PENDING
        )


def model_ref(category: str, name: str) -> str:
    """Return a stable reference to a named Model IR element."""
    return f"model:{category}:{name}"


def analysis_ref(analysis: str, *parts: str) -> str:
    """Return a stable reference to a deterministic analysis result."""
    suffix = ":".join(parts)
    return f"analysis:{analysis}" + (f":{suffix}" if suffix else "")
