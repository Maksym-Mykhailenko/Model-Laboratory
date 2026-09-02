"""Portable semantics for reproducing stochastic experiment results.

Stochastic equality is deliberately separate from deterministic numerical reproduction.
An identical sample stream can reproduce exactly.  Independently generated samples may
instead be statistically equivalent under criteria frozen before reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

import numpy as np
from scipy.stats import ks_2samp

from .canonical import canonical_json_sha256
from .experiment import (
    ExperimentStateError,
    _array_fingerprint,
    _canonical_sample_indices,
    _decode_array_reference,
    _encode_array_reference,
)


STOCHASTIC_REFERENCE_SCHEMA = "model-laboratory-stochastic-reference"
STOCHASTIC_REFERENCE_SCHEMA_VERSION = "1.0"


class StochasticReproductionError(ValueError):
    """Raised when stochastic settings or reference data are invalid."""


class StochasticReproductionStatus(str, Enum):
    """Scientific outcomes available for one stochastic result."""

    EXACT = "EXACT STOCHASTIC REPRODUCTION"
    STATISTICALLY_EQUIVALENT = "STATISTICALLY EQUIVALENT"
    NOT_EQUIVALENT = "NOT STATISTICALLY EQUIVALENT"
    UNABLE = "UNABLE TO ASSESS STOCHASTIC REPRODUCTION"


@dataclass(frozen=True, slots=True)
class RNGSpecification:
    """The complete random-number generator identity used by a result."""

    seed: int
    algorithm: str
    algorithm_version: str
    implementation: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**128
        ):
            raise StochasticReproductionError("seed must be an integer in [0, 2**128).")
        for field, value in (
            ("algorithm", self.algorithm),
            ("algorithm_version", self.algorithm_version),
            ("implementation", self.implementation),
        ):
            if not isinstance(value, str) or not value.strip():
                raise StochasticReproductionError(f"{field} must be a non-empty string.")

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "implementation": self.implementation,
        }


@dataclass(frozen=True, slots=True)
class SamplingConfiguration:
    """Frozen sampling and chain layout for a stochastic result."""

    chains: int
    draws_per_chain: int
    warmup_draws_per_chain: int = 0
    settings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.chains, bool)
            or not isinstance(self.chains, int)
            or self.chains < 1
        ):
            raise StochasticReproductionError("chains must be a positive integer.")
        if (
            isinstance(self.draws_per_chain, bool)
            or not isinstance(self.draws_per_chain, int)
            or self.draws_per_chain < 1
        ):
            raise StochasticReproductionError("draws_per_chain must be a positive integer.")
        if (
            isinstance(self.warmup_draws_per_chain, bool)
            or not isinstance(self.warmup_draws_per_chain, int)
            or self.warmup_draws_per_chain < 0
        ):
            raise StochasticReproductionError(
                "warmup_draws_per_chain must be a non-negative integer."
            )
        names: set[str] = set()
        canonical: list[tuple[str, str]] = []
        for item in self.settings:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or not isinstance(item[1], str)
            ):
                raise StochasticReproductionError(
                    "sampling settings must be (non-empty name, string value) pairs."
                )
            if item[0] in names:
                raise StochasticReproductionError(
                    f"sampling setting names must be unique: {item[0]!r}."
                )
            names.add(item[0])
            canonical.append(item)
        object.__setattr__(self, "settings", tuple(sorted(canonical)))

    @property
    def sample_count(self) -> int:
        return self.chains * self.draws_per_chain

    def to_dict(self) -> dict[str, object]:
        return {
            "chains": self.chains,
            "draws_per_chain": self.draws_per_chain,
            "warmup_draws_per_chain": self.warmup_draws_per_chain,
            "sample_count": self.sample_count,
            "settings": [
                {"name": name, "value": value} for name, value in self.settings
            ],
        }


@dataclass(frozen=True, slots=True)
class StochasticEquivalenceSettings:
    """Pre-declared criteria for statistical, rather than exact, equivalence."""

    significance_level: float = 0.01
    mean_relative_tolerance: float = 0.05
    mean_absolute_tolerance: float = 0.02
    variance_relative_tolerance: float = 0.10
    variance_absolute_tolerance: float = 0.02
    minimum_sample_count: int = 200

    def __post_init__(self) -> None:
        values = (
            self.significance_level,
            self.mean_relative_tolerance,
            self.mean_absolute_tolerance,
            self.variance_relative_tolerance,
            self.variance_absolute_tolerance,
        )
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
            raise StochasticReproductionError("equivalence tolerances must be finite numbers.")
        if not 0 < self.significance_level < 1:
            raise StochasticReproductionError("significance_level must be between zero and one.")
        if any(value < 0 for value in values[1:]):
            raise StochasticReproductionError("equivalence tolerances cannot be negative.")
        if (
            isinstance(self.minimum_sample_count, bool)
            or not isinstance(self.minimum_sample_count, int)
            or self.minimum_sample_count < 2
        ):
            raise StochasticReproductionError("minimum_sample_count must be at least two.")

    def to_dict(self) -> dict[str, object]:
        return {
            "significance_level": self.significance_level,
            "mean_relative_tolerance": self.mean_relative_tolerance,
            "mean_absolute_tolerance": self.mean_absolute_tolerance,
            "variance_relative_tolerance": self.variance_relative_tolerance,
            "variance_absolute_tolerance": self.variance_absolute_tolerance,
            "minimum_sample_count": self.minimum_sample_count,
        }


@dataclass(frozen=True, slots=True)
class StochasticSampleResult:
    """One scalar stochastic result with explicit chain and RNG metadata."""

    name: str
    samples: np.ndarray
    rng: RNGSpecification
    sampling: SamplingConfiguration

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise StochasticReproductionError("A stochastic result name is required.")
        values = np.asarray(self.samples, dtype="<f8")
        if values.size != self.sampling.sample_count:
            raise StochasticReproductionError(
                "sample count does not match chains multiplied by draws_per_chain."
            )
        if not np.all(np.isfinite(values)):
            raise StochasticReproductionError("stochastic samples must all be finite.")
        values = values.reshape(self.sampling.chains, self.sampling.draws_per_chain).copy()
        values[values == 0.0] = 0.0
        values.setflags(write=False)
        object.__setattr__(self, "samples", values)


@dataclass(frozen=True, slots=True)
class StochasticComparison:
    """Structured stochastic reproduction evidence."""

    status: StochasticReproductionStatus
    exact_fingerprint_matches: bool
    rng_matches: bool
    sampling_configuration_matches: bool
    saved_sample_count: int
    current_sample_count: int
    mean_absolute_deviation: float | None = None
    variance_absolute_deviation: float | None = None
    kolmogorov_smirnov_statistic: float | None = None
    kolmogorov_smirnov_p_value: float | None = None
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "exact_fingerprint_matches": self.exact_fingerprint_matches,
            "rng_matches": self.rng_matches,
            "sampling_configuration_matches": self.sampling_configuration_matches,
            "saved_sample_count": self.saved_sample_count,
            "current_sample_count": self.current_sample_count,
            "mean_absolute_deviation": self.mean_absolute_deviation,
            "variance_absolute_deviation": self.variance_absolute_deviation,
            "kolmogorov_smirnov_statistic": self.kolmogorov_smirnov_statistic,
            "kolmogorov_smirnov_p_value": self.kolmogorov_smirnov_p_value,
            "details": list(self.details),
        }


def _summary(samples: np.ndarray) -> dict[str, object]:
    flat = np.asarray(samples, dtype=np.float64).reshape(-1)
    return {
        "sample_count": int(flat.size),
        "mean": float(np.mean(flat)),
        "variance": float(np.var(flat, ddof=1)),
        "quantiles": {
            "0.05": float(np.quantile(flat, 0.05)),
            "0.50": float(np.quantile(flat, 0.50)),
            "0.95": float(np.quantile(flat, 0.95)),
        },
    }


def stochastic_result_fingerprint(result: StochasticSampleResult) -> str:
    """Return strict identity for the RNG, sampling contract and complete sample stream."""
    return canonical_json_sha256(
        {
            "name": result.name,
            "rng": result.rng.to_dict(),
            "sampling": result.sampling.to_dict(),
            "sample_shape": list(result.samples.shape),
            "sample_sha256": _array_fingerprint(result.samples),
        }
    )


def stochastic_reference(
    result: StochasticSampleResult,
    settings: StochasticEquivalenceSettings,
) -> dict[str, object]:
    """Create a portable reference with criteria frozen before later comparison."""
    return {
        "schema": STOCHASTIC_REFERENCE_SCHEMA,
        "schema_version": STOCHASTIC_REFERENCE_SCHEMA_VERSION,
        "name": result.name,
        "rng": result.rng.to_dict(),
        "sampling": result.sampling.to_dict(),
        "exact_sha256": stochastic_result_fingerprint(result),
        "samples": _encode_array_reference(result.samples.reshape(-1)),
        "summary": _summary(result.samples),
        "statistical_equivalence_settings": settings.to_dict(),
        "claim": "statistical equivalence is distinct from deterministic numerical reproduction",
    }


def _saved_sample_values(reference: Mapping[str, Any], current: np.ndarray) -> np.ndarray:
    sample_reference = reference.get("samples")
    if not isinstance(sample_reference, Mapping):
        raise StochasticReproductionError("saved stochastic samples are malformed.")
    encoding = sample_reference.get("encoding")
    try:
        if encoding != "canonical-samples-v1":
            return _decode_array_reference(sample_reference).reshape(-1)
        values_raw = sample_reference.get("sample_values")
        indices_raw = sample_reference.get("sample_indices")
        if not isinstance(values_raw, Mapping) or not isinstance(indices_raw, list):
            raise StochasticReproductionError("saved stochastic sample selection is malformed.")
        saved = _decode_array_reference(values_raw).reshape(-1)
        if saved.size != len(indices_raw):
            raise StochasticReproductionError("saved stochastic sample selection is malformed.")
        return saved
    except ExperimentStateError as exc:
        raise StochasticReproductionError(str(exc)) from exc


def _current_comparison_values(reference: Mapping[str, Any], current: np.ndarray) -> np.ndarray:
    sample_reference = reference["samples"]
    current_flat = np.asarray(current, dtype=np.float64).reshape(-1)
    if sample_reference.get("encoding") != "canonical-samples-v1":
        return current_flat
    saved_count = len(sample_reference.get("sample_indices", ()))
    indices = _canonical_sample_indices(int(current_flat.size))
    if len(indices) > saved_count:
        indices = [
            (index * (current_flat.size - 1)) // (saved_count - 1)
            for index in range(saved_count)
        ]
    return current_flat[indices]


def _isclose(current: float, saved: float, *, rtol: float, atol: float) -> bool:
    return bool(np.isclose(current, saved, rtol=rtol, atol=atol))


def compare_stochastic_result(
    reference: Mapping[str, Any],
    current: StochasticSampleResult,
) -> StochasticComparison:
    """Compare exact sample identity, then assess the frozen statistical contract."""
    try:
        if (
            reference.get("schema") != STOCHASTIC_REFERENCE_SCHEMA
            or reference.get("schema_version") != STOCHASTIC_REFERENCE_SCHEMA_VERSION
        ):
            raise StochasticReproductionError("unsupported stochastic reference schema.")
        settings_raw = reference["statistical_equivalence_settings"]
        if not isinstance(settings_raw, Mapping):
            raise StochasticReproductionError("stochastic equivalence settings are malformed.")
        settings = StochasticEquivalenceSettings(**dict(settings_raw))
        saved_summary = reference["summary"]
        if not isinstance(saved_summary, Mapping):
            raise StochasticReproductionError("saved stochastic summary is malformed.")
        saved_count = int(saved_summary["sample_count"])
        exact = reference.get("exact_sha256") == stochastic_result_fingerprint(current)
        rng_matches = reference.get("rng") == current.rng.to_dict()
        sampling_matches = reference.get("sampling") == current.sampling.to_dict()
        if exact:
            return StochasticComparison(
                StochasticReproductionStatus.EXACT,
                True,
                rng_matches,
                sampling_matches,
                saved_count,
                current.samples.size,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        if min(saved_count, current.samples.size) < settings.minimum_sample_count:
            return StochasticComparison(
                StochasticReproductionStatus.UNABLE,
                False,
                rng_matches,
                sampling_matches,
                saved_count,
                current.samples.size,
                details=("sample count is below the frozen minimum for statistical assessment",),
            )

        saved_values = _saved_sample_values(reference, current.samples)
        current_values = _current_comparison_values(reference, current.samples)
        if min(saved_values.size, current_values.size) < settings.minimum_sample_count:
            raise StochasticReproductionError(
                "portable sample selection is below the frozen minimum sample count."
            )
        current_summary = _summary(current.samples)
        saved_mean = float(saved_summary["mean"])
        saved_variance = float(saved_summary["variance"])
        current_mean = float(current_summary["mean"])
        current_variance = float(current_summary["variance"])
        mean_matches = _isclose(
            current_mean,
            saved_mean,
            rtol=settings.mean_relative_tolerance,
            atol=settings.mean_absolute_tolerance,
        )
        variance_matches = _isclose(
            current_variance,
            saved_variance,
            rtol=settings.variance_relative_tolerance,
            atol=settings.variance_absolute_tolerance,
        )
        ks = ks_2samp(saved_values, current_values, method="auto")
        distribution_matches = bool(float(ks.pvalue) >= settings.significance_level)
        matches = mean_matches and variance_matches and distribution_matches
        details: list[str] = []
        if not mean_matches:
            details.append("sample mean exceeds the frozen tolerance")
        if not variance_matches:
            details.append("sample variance exceeds the frozen tolerance")
        if not distribution_matches:
            details.append("two-sample KS test rejects the frozen distributional criterion")
        if not rng_matches:
            details.append("RNG identity differs; only statistical equivalence can be claimed")
        if not sampling_matches:
            details.append("sampling configuration differs; only statistical equivalence can be claimed")
        return StochasticComparison(
            StochasticReproductionStatus.STATISTICALLY_EQUIVALENT
            if matches
            else StochasticReproductionStatus.NOT_EQUIVALENT,
            False,
            rng_matches,
            sampling_matches,
            saved_count,
            current.samples.size,
            abs(current_mean - saved_mean),
            abs(current_variance - saved_variance),
            float(ks.statistic),
            float(ks.pvalue),
            tuple(details),
        )
    except (KeyError, TypeError, ValueError, OverflowError, StochasticReproductionError) as exc:
        return StochasticComparison(
            StochasticReproductionStatus.UNABLE,
            False,
            False,
            False,
            0,
            current.samples.size,
            details=(str(exc),),
        )
