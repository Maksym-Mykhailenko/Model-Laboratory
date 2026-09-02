from __future__ import annotations

import numpy as np

from model_lab.stochastic import (
    RNGSpecification,
    SamplingConfiguration,
    StochasticEquivalenceSettings,
    StochasticReproductionStatus,
    StochasticSampleResult,
    compare_stochastic_result,
    stochastic_reference,
)


def _result(seed: int, *, location: float = 0.0, draws: int = 10_000) -> StochasticSampleResult:
    rng = np.random.Generator(np.random.PCG64(seed))
    return StochasticSampleResult(
        name="posterior.theta",
        samples=rng.normal(location, 1.0, draws),
        rng=RNGSpecification(
            seed=seed,
            algorithm="PCG64",
            algorithm_version=np.__version__,
            implementation="numpy.random.Generator",
        ),
        sampling=SamplingConfiguration(
            chains=2,
            draws_per_chain=draws // 2,
            warmup_draws_per_chain=500,
            settings=(("sampler", "independent-normal"),),
        ),
    )


def test_stochastic_reference_records_rng_chain_and_frozen_equivalence_contract() -> None:
    result = _result(7)
    settings = StochasticEquivalenceSettings(significance_level=0.001)
    reference = stochastic_reference(result, settings)

    assert reference["rng"]["seed"] == 7
    assert reference["rng"]["algorithm"] == "PCG64"
    assert reference["sampling"]["chains"] == 2
    assert reference["sampling"]["warmup_draws_per_chain"] == 500
    assert reference["statistical_equivalence_settings"] == settings.to_dict()


def test_identical_stochastic_stream_is_exact() -> None:
    result = _result(7)
    comparison = compare_stochastic_result(
        stochastic_reference(result, StochasticEquivalenceSettings()), result
    )

    assert comparison.status is StochasticReproductionStatus.EXACT
    assert comparison.exact_fingerprint_matches
    assert comparison.rng_matches
    assert comparison.sampling_configuration_matches


def test_independent_equivalent_stream_gets_statistical_not_numerical_claim() -> None:
    saved = _result(7)
    current = _result(11)
    settings = StochasticEquivalenceSettings(
        significance_level=1e-6,
        mean_absolute_tolerance=0.05,
        variance_absolute_tolerance=0.05,
    )
    comparison = compare_stochastic_result(stochastic_reference(saved, settings), current)

    assert comparison.status is StochasticReproductionStatus.STATISTICALLY_EQUIVALENT
    assert not comparison.exact_fingerprint_matches
    assert not comparison.rng_matches
    assert comparison.kolmogorov_smirnov_p_value is not None


def test_materially_different_distribution_is_not_equivalent() -> None:
    saved = _result(7)
    current = _result(11, location=1.0)
    comparison = compare_stochastic_result(
        stochastic_reference(saved, StochasticEquivalenceSettings()), current
    )

    assert comparison.status is StochasticReproductionStatus.NOT_EQUIVALENT
    assert comparison.details


def test_insufficient_samples_are_unable_not_conflicting() -> None:
    saved = _result(7, draws=100)
    current = _result(11, draws=100)
    comparison = compare_stochastic_result(
        stochastic_reference(
            saved, StochasticEquivalenceSettings(minimum_sample_count=200)
        ),
        current,
    )

    assert comparison.status is StochasticReproductionStatus.UNABLE
