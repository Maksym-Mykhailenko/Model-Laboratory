from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pytest

from model_lab.interpreter import (
    INTERPRETER_OUTPUT_SCHEMA,
    INTERPRETER_OUTPUT_SCHEMA_VERSION,
    InterpreterProposalError,
    _official_kind_context_selection,
    _official_pack_context_selection,
    create_interpreter_context,
    process_interpreter_output,
    scientific_request_boundary,
)
from model_lab.parser import parse_model_text
from model_lab.validator import validate_model

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "verification" / "interpreter_baseline_v1.6.json"
CORPUS = ROOT / "training" / "interpreter_corpus_v1.5"

OFFICIAL_FAMILIES = {
    "official-multidimensional",
    "official-probability",
    "official-graphs",
    "official-generative",
    "official-dynamics",
    "official-fields",
    "official-geometry",
    "official-mechanics",
    "official-statistics",
    "official-optimisation",
    "official-electrical",
    "official-reactions",
    "official-learning",
}


def _expected_source(case: dict) -> str:
    expected = case["expected"]
    source = expected.get("model_source")
    if source:
        return source
    return (expected.get("clarification_followup") or {}).get("model_source", "")


def _records(split: str) -> list[dict]:
    path = CORPUS / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _proposal_semantics(record: dict) -> str | None:
    if record["target"].get("action") != "propose_edits":
        return None
    return record["compiled"]["scientific_semantics_sha256"]


def test_every_official_benchmark_target_is_exposed_by_its_actual_prompt_context() -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    checked = 0
    for case in benchmark["cases"]:
        if case["family"] not in OFFICIAL_FAMILIES:
            continue
        source = _expected_source(case)
        if not source:
            continue
        graph = validate_model(parse_model_text(source)).graph
        expected_kinds = {item.kind for item in graph.objects}
        context = create_interpreter_context(
            instruction=case["instruction"],
            current_model_source=case.get("current_model_source", ""),
        )
        exposed = {
            item["kind"] for item in context["contract"]["installed_model_object_kinds"]
        }
        assert expected_kinds <= exposed, (case["id"], expected_kinds - exposed)
        checked += 1
    assert checked == 60


def test_router_recognises_ordinary_scientific_aliases() -> None:
    cases = {
        "Create a finite distribution over three outcomes.": "org.modellab.pack.probability-stochastic-systems",
        "Model a damped spring oscillator.": "org.modellab.pack.dynamics-differential-equations-control",
        "Create a tetrahedral mesh of this shape.": "org.modellab.pack.geometry-meshes-spatial-computation",
        "Set up a regression study for these measurements.": "org.modellab.pack.statistical-inference-data-modelling",
        "Minimise this nonlinear objective subject to bounds.": "org.modellab.pack.optimisation-estimation-inverse-problems",
        "Build a resistor divider with a voltage source.": "org.modellab.pack.electrical-electronic-electromagnetic-systems",
        "Create a two compartment pharmacokinetic model.": "org.modellab.pack.chemical-reaction-biological-systems",
        "Create a feedforward neural classifier.": "org.modellab.pack.machine-learning-computational-intelligence",
    }
    for instruction, expected in cases.items():
        selection = _official_pack_context_selection(instruction, None)
        assert expected in selection, (instruction, selection)


def test_official_corpus_is_diverse_split_isolated_and_not_schema_copying() -> None:
    train = [r for r in _records("train") if r["family"] in OFFICIAL_FAMILIES]
    validation = [r for r in _records("validation") if r["family"] in OFFICIAL_FAMILIES]
    records = train + validation
    assert len(records) == 1300

    instructions = [r["conversation"]["instruction"].casefold() for r in records]
    assert all("org.modellab" not in text for text in instructions)
    assert all("properties exactly" not in text for text in instructions)
    assert any(r["conversation"].get("current_model_source") for r in records)
    assert {r["target"]["action"] for r in records} >= {"propose_edits", "needs_clarification"}

    conversations: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        cid = record["conversation"].get("conversation_id")
        if cid:
            conversations[cid].append(record)
    paired = [
        turns
        for turns in conversations.values()
        if {int(t["conversation"].get("turn_index", -1)) for t in turns} == {0, 1}
        and any(t["target"].get("action") == "needs_clarification" for t in turns)
        and any(t["target"].get("action") == "propose_edits" for t in turns)
    ]
    assert len(paired) >= 250

    archetypes = {(r["family"], r.get("archetype")) for r in records}
    assert len(archetypes) >= 200

    train_semantics = {s for r in train if (s := _proposal_semantics(r))}
    val_semantics = {s for r in validation if (s := _proposal_semantics(r))}
    assert len(train_semantics) >= 700
    assert len(val_semantics) >= 140
    assert train_semantics.isdisjoint(val_semantics)


def test_benchmark_has_creation_edit_and_clarification_for_every_official_family() -> None:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    by_family: dict[str, list[dict]] = defaultdict(list)
    for case in benchmark["cases"]:
        if case["family"] in OFFICIAL_FAMILIES:
            by_family[case["family"]].append(case)
    assert set(by_family) == OFFICIAL_FAMILIES
    for family, cases in by_family.items():
        ids = [case["id"] for case in cases]
        assert any(case_id.endswith("-edit") for case_id in ids), family
        assert any(case_id.endswith("-clarify") for case_id in ids), family
        assert any(
            not case_id.endswith("-edit") and not case_id.endswith("-clarify")
            for case_id in ids
        ), family


def test_router_handles_plural_precision_and_multi_pack_budget() -> None:
    expected = {
        "Create two matrices.": "org.modellab.pack.multidimensional-mathematics",
        "Compare probability distributions.": "org.modellab.pack.probability-stochastic-systems",
        "Analyse several graphs.": "org.modellab.pack.graphs-networks-discrete",
        "Build two HMMs.": "org.modellab.pack.generative-inference-decision-systems",
        "Solve these ODEs.": "org.modellab.pack.dynamics-differential-equations-control",
        "Set up coupled PDEs.": "org.modellab.pack.spatial-fields-continuum-pdes",
        "Compare two trusses.": "org.modellab.pack.mechanics-structures-materials",
        "Train classifiers.": "org.modellab.pack.machine-learning-computational-intelligence",
        "Simulate chemical reactions.": "org.modellab.pack.chemical-reaction-biological-systems",
    }
    for instruction, pack in expected.items():
        assert pack in _official_pack_context_selection(instruction, None)

    neural = _official_pack_context_selection("Create a neural network.", None)
    assert neural == ("org.modellab.pack.machine-learning-computational-intelligence",)
    poisson_ratio = _official_pack_context_selection("Use a Poisson ratio of 0.3.", None)
    assert "org.modellab.pack.mechanics-structures-materials" in poisson_ratio
    assert "org.modellab.pack.spatial-fields-continuum-pdes" not in poisson_ratio
    reaction = _official_pack_context_selection("Create a reaction network.", None)
    assert "org.modellab.pack.chemical-reaction-biological-systems" in reaction
    assert "org.modellab.pack.graphs-networks-discrete" not in reaction

    context = create_interpreter_context(
        instruction="Create a neural controller for an ODE system.", current_model_source=""
    )
    exposed = {item["kind"] for item in context["contract"]["installed_model_object_kinds"]}
    assert any(kind.startswith("org.modellab.dynamics.") for kind in exposed)
    assert any(kind.startswith("org.modellab.learning.") for kind in exposed)


def test_review_queue_is_conversation_atomic_and_covers_every_official_archetype() -> None:
    records = _records("train") + _records("validation")
    by_id = {record["id"]: record for record in records}
    review = [
        json.loads(line)
        for line in (CORPUS / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    reviewed = {item["id"] for item in review}
    conversations: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["family"] in OFFICIAL_FAMILIES:
            conversations[record["conversation"]["conversation_id"]].append(record)
    for turns in conversations.values():
        if any(turn["target"]["action"] == "needs_clarification" for turn in turns):
            flags = [turn["id"] in reviewed for turn in turns]
            assert not any(flags) or all(flags)

    all_archetypes = {
        (r["family"], r["split"], r["archetype"])
        for r in records if r["family"] in OFFICIAL_FAMILIES
    }
    reviewed_archetypes = {
        (by_id[item["id"]]["family"], by_id[item["id"]]["split"], by_id[item["id"]]["archetype"])
        for item in review if by_id[item["id"]]["family"] in OFFICIAL_FAMILIES
    }
    assert all_archetypes <= reviewed_archetypes


def test_optimisation_corpus_uses_decision_variable_language() -> None:
    records = [
        r for r in (_records("train") + _records("validation"))
        if r["family"] == "official-optimisation"
    ]
    assert records
    assert all("initial probabilities" not in r["conversation"]["instruction"].casefold() for r in records)


def test_object_kind_routing_covers_cross_pack_language_without_network_overrouting() -> None:
    from model_lab.interpreter import _official_kind_context_selection

    expectations = {
        "Build an RLC network": {
            "org.modellab.electrical.linear-circuit",
        },
        "Optimize a truss": {
            "org.modellab.mechanics.truss-structure",
            "org.modellab.optimisation.nonlinear-problem",
        },
        "Estimate the parameters of this ODE": {
            "org.modellab.dynamics.ode-system",
            "org.modellab.estimation.nonlinear-least-squares",
        },
        "Solve Laplace equation on a mesh": {
            "org.modellab.pde.poisson-problem",
            "org.modellab.geometry.triangle-mesh",
        },
        "Create an LTI controller": {
            "org.modellab.control.state-space-system",
        },
    }
    for instruction, expected in expectations.items():
        assert set(_official_kind_context_selection(instruction, None)) == expected

    graph_pack = "org.modellab.pack.graphs-networks-discrete"
    for instruction in ("Build a resistor network", "Create a circuit network"):
        assert graph_pack not in _official_pack_context_selection(instruction, None)
    assert _official_pack_context_selection("Create a graph neural network.", None) == (
        "org.modellab.pack.machine-learning-computational-intelligence",
    )


def test_context_budget_fallback_accounts_for_attachment_and_history_overhead() -> None:
    import base64
    from model_lab.interpreter import MAX_CONTEXT_PACKAGE_BYTES, _canonical_json_bytes
    from model_lab.interpreter_attachments import ingest_attachment

    attachment = ingest_attachment(
        "note.txt", base64.b64encode(b"tiny reference").decode("ascii")
    )
    history = [
        {"question": "q" * 900, "answer": "a" * 900}
        for _ in range(8)
    ]
    for instruction in (
        "Create a graph neural network.",
        "Build a neural controller for an ODE model of a reaction network.",
    ):
        context = create_interpreter_context(
            instruction=instruction,
            current_model_source="",
            attachments=[attachment],
            clarification_history=history,
        )
        assert len(_canonical_json_bytes(context)) <= MAX_CONTEXT_PACKAGE_BYTES
        assert context["attachments"]["documents"]
        assert context["contract"]["installed_capabilities"] == []
        assert context["contract"]["model"]["generated_schema_catalogue"][
            "omitted_for_context_budget"
        ] is True


def test_review_queue_preserves_core_sample_and_uses_only_required_official_rows() -> None:
    records = _records("train") + _records("validation")
    review = [
        json.loads(line)
        for line in (CORPUS / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    by_id = {record["id"]: record for record in records}
    official = [item for item in review if by_id[item["id"]]["family"] in OFFICIAL_FAMILIES]
    core = [item for item in review if by_id[item["id"]]["family"] not in OFFICIAL_FAMILIES]
    assert len(review) == 725
    assert len(official) == 306
    assert len(core) == 419


def test_official_training_language_has_no_identifier_directive_or_duplicate_model_name() -> None:
    records = [
        r for r in (_records("train") + _records("validation"))
        if r["family"] in OFFICIAL_FAMILIES
    ]
    assert all("use these exact identifiers" not in r["conversation"]["instruction"].casefold() for r in records)
    for record in records:
        if record["target"].get("action") != "propose_edits" or record["conversation"].get("current_model_source"):
            continue
        name_ops = [
            op for op in record["target"].get("operations", [])
            if op.get("op") == "set" and op.get("path") == ["name"] and isinstance(op.get("value"), str)
        ]
        if not name_ops:
            continue
        model_name = name_ops[0]["value"].casefold()
        assert record["conversation"]["instruction"].casefold().count(model_name) == 1


def test_mixed_specificity_object_kind_routing_is_compositional() -> None:
    cases = {
        "Build a truss and a machine learning model.": {
            "org.modellab.mechanics.truss-structure",
            "org.modellab.intelligence.fuzzy-rule-system",
            "org.modellab.learning.feature-dataset",
            "org.modellab.learning.feedforward-network",
            "org.modellab.learning.supervised-study",
        },
        "Create an ODE and a statistical model.": {
            "org.modellab.dynamics.ode-system",
            "org.modellab.statistics.dataset",
            "org.modellab.statistics.grouped-samples",
            "org.modellab.statistics.linear-model-study",
        },
        "Create a graph and a continuum model.": {
            "org.modellab.graph.network",
            "org.modellab.field.structured-scalar-field",
            "org.modellab.field.structured-vector-field",
            "org.modellab.pde.diffusion-problem",
            "org.modellab.pde.poisson-problem",
        },
    }
    for instruction, expected in cases.items():
        context = create_interpreter_context(
            instruction=instruction, current_model_source=""
        )
        exposed = {
            item["kind"]
            for item in context["contract"]["installed_model_object_kinds"]
        }
        assert expected <= exposed, (instruction, expected - exposed)

    broad_ml = create_interpreter_context(
        instruction="Create a machine learning model.", current_model_source=""
    )
    official = {
        item["kind"]
        for item in broad_ml["contract"]["installed_model_object_kinds"]
        if item["kind"].startswith(("org.modellab.learning.", "org.modellab.intelligence."))
    }
    assert official == {
        "org.modellab.intelligence.fuzzy-rule-system",
        "org.modellab.learning.feature-dataset",
        "org.modellab.learning.feedforward-network",
        "org.modellab.learning.supervised-study",
    }


def test_router_covers_remaining_ordinary_scientific_language() -> None:
    from model_lab.interpreter import _official_kind_context_selection

    expectations = {
        "Create a network with 10 nodes and 20 edges.": {"org.modellab.graph.network"},
        "Build an undirected network.": {"org.modellab.graph.network"},
        "Create a categorical distribution.": {"org.modellab.probability.discrete-distribution"},
        "Create a Bernoulli distribution.": {"org.modellab.probability.discrete-distribution"},
        "Fit an ODE model to data.": {
            "org.modellab.dynamics.ode-system",
            "org.modellab.estimation.nonlinear-least-squares",
        },
        "Calibrate the parameters of this ODE.": {
            "org.modellab.dynamics.ode-system",
            "org.modellab.estimation.nonlinear-least-squares",
        },
        "Identify the parameters of this ODE.": {
            "org.modellab.dynamics.ode-system",
            "org.modellab.estimation.nonlinear-least-squares",
        },
        "Create an electrical network.": {"org.modellab.electrical.linear-circuit"},
        "Create a mass spring system.": {"org.modellab.dynamics.ode-system"},
    }
    for instruction, expected in expectations.items():
        selected = set(_official_kind_context_selection(instruction, None))
        assert expected <= selected, (instruction, expected - selected)


def test_combined_attachment_and_max_history_budget_is_end_to_end() -> None:
    import base64
    from model_lab.canonical import canonical_json_sha256
    from model_lab.interpreter import MAX_CONTEXT_PACKAGE_BYTES, _canonical_json_bytes
    from model_lab.interpreter_attachments import ingest_attachment

    attachments = [
        ingest_attachment(
            f"a{index}_" + "x" * 190 + ".txt",
            base64.b64encode(b"x").decode("ascii"),
        )
        for index in range(8)
    ]
    history = [
        {"question": "q" * 1000, "answer": "a" * 1000}
        for _ in range(8)
    ]
    assert len(_canonical_json_bytes(history)) == 16225
    context = create_interpreter_context(
        instruction="Create a machine learning model.",
        current_model_source="",
        attachments=attachments,
        clarification_history=history,
    )
    assert len(_canonical_json_bytes(context)) <= MAX_CONTEXT_PACKAGE_BYTES
    assert len(context["clarification_history"]) == len(history)
    assert context["clarification_history_sha256"] == canonical_json_sha256(history)
    # This exact accepted-limit reproduction fits without discarding any turn content.
    assert context["clarification_history"] == history

    mixed = create_interpreter_context(
        instruction="Build a truss and a machine learning model.",
        current_model_source="",
        attachments=attachments,
        clarification_history=history,
    )
    assert len(_canonical_json_bytes(mixed)) <= MAX_CONTEXT_PACKAGE_BYTES
    assert len(mixed["clarification_history"]) == len(history)
    assert mixed["clarification_history_sha256"] == canonical_json_sha256(history)
    assert mixed["clarification_history"][0]["question"].startswith(
        "[older clarification question omitted for context budget;"
    )
    assert mixed["clarification_history"][1:] == history[1:]


def test_router_covers_remaining_scientific_aliases_and_negation() -> None:
    from model_lab.interpreter import _official_kind_context_selection

    expectations = {
        "Maximise this objective.": {"org.modellab.optimisation.nonlinear-problem"},
        "Create an MLP.": {"org.modellab.learning.feedforward-network"},
        "Create a multilayer perceptron.": {"org.modellab.learning.feedforward-network"},
        "Infer the parameters of this ODE.": {
            "org.modellab.dynamics.ode-system",
            "org.modellab.estimation.nonlinear-least-squares",
        },
        "Create a network with 10 vertices and 20 links.": {"org.modellab.graph.network"},
        "Solve Laplace's equation on a mesh.": {
            "org.modellab.pde.poisson-problem",
            "org.modellab.geometry.triangle-mesh",
        },
        "Create a compartmental model.": {"org.modellab.biological.compartment-system"},
        "Create a chemical kinetics model.": {"org.modellab.chemistry.mass-action-network"},
    }
    for instruction, expected in expectations.items():
        assert expected <= set(_official_kind_context_selection(instruction, None)), instruction

    assert set(_official_kind_context_selection(
        "Do not optimise the truss; just analyse it.", None
    )) == {"org.modellab.mechanics.truss-structure"}
    assert set(_official_kind_context_selection(
        "This is not a graph; create a reaction network.", None
    )) == {"org.modellab.chemistry.mass-action-network"}
    assert set(_official_kind_context_selection(
        "Create an ODE without parameter estimation.", None
    )) == {"org.modellab.dynamics.ode-system"}


def test_existing_mixed_pack_model_does_not_inject_every_schema_for_model_level_edit() -> None:
    import base64
    import yaml

    from model_lab.interpreter import MAX_CONTEXT_PACKAGE_BYTES, _canonical_json_bytes
    from model_lab.interpreter_attachments import ingest_attachment

    objects: dict[str, object] = {}
    for path in sorted((ROOT / "models").glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for identifier, value in (document.get("objects") or {}).items():
            assert identifier not in objects
            objects[identifier] = value
    assert len(objects) >= 26

    source = yaml.safe_dump(
        {"name": "rich-model", "objects": objects},
        sort_keys=False,
        allow_unicode=True,
        width=1000,
    )
    validate_model(parse_model_text(source))
    attachment = ingest_attachment(
        "x.txt", base64.b64encode(b"x").decode("ascii")
    )
    context = create_interpreter_context(
        instruction="Rename the model.",
        current_model_source=source,
        attachments=[attachment],
    )
    assert len(_canonical_json_bytes(context)) <= MAX_CONTEXT_PACKAGE_BYTES
    assert not {
        item["kind"] for item in context["contract"]["installed_model_object_kinds"]
        if str(item["kind"]).startswith("org.modellab.")
        and not str(item["kind"]).startswith("org.modellab.core.")
    }


def test_existing_object_identifier_routes_only_its_kind_in_rich_model() -> None:
    import yaml
    from model_lab.interpreter import _official_kind_context_selection

    objects: dict[str, object] = {}
    for path in sorted((ROOT / "models").glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        objects.update(document.get("objects") or {})
    source = yaml.safe_dump({"name": "rich-model", "objects": objects}, sort_keys=False)
    model = validate_model(parse_model_text(source))
    assert _official_kind_context_selection("Edit roof-truss.", model) == (
        "org.modellab.mechanics.truss-structure",
    )


def test_router_handles_extended_negation_scopes_and_contractions() -> None:
    optimisation = "org.modellab.pack.optimisation-estimation-inverse-problems"
    graph = "org.modellab.pack.graphs-networks-discrete"
    mechanics = "org.modellab.pack.mechanics-structures-materials"
    dynamics = "org.modellab.pack.dynamics-differential-equations-control"
    reactions = "org.modellab.pack.chemical-reaction-biological-systems"

    for instruction in (
        "Don't optimise the truss",
        "Never optimise the truss",
        "Do not try to optimise the truss",
    ):
        packs = _official_pack_context_selection(instruction, None)
        assert mechanics in packs
        assert optimisation not in packs

    for instruction in (
        "No parameter estimation; create an ODE",
        "Create an ODE but do not attempt parameter estimation",
    ):
        packs = _official_pack_context_selection(instruction, None)
        assert dynamics in packs
        assert optimisation not in packs

    for instruction in (
        "This isn't a graph; create a reaction network",
        "Avoid graph analysis; create a reaction network",
        "Do not create a graph; create a reaction network",
    ):
        packs = _official_pack_context_selection(instruction, None)
        assert reactions in packs
        assert graph not in packs


def test_router_covers_remaining_common_scientific_language() -> None:
    from model_lab.interpreter import _official_kind_context_selection

    cases = {
        "Create a binary classification model.": {"org.modellab.learning.supervised-study"},
        "Build a harmonic oscillator.": {"org.modellab.dynamics.ode-system"},
        "Set the elastic modulus to 200 GPa.": {"org.modellab.material.isotropic-linear-elastic"},
        "Add a capacitor and an inductor.": {"org.modellab.electrical.linear-circuit"},
        "Model Coulomb forces between charges.": {"org.modellab.electromagnetics.point-charge-system"},
        "Fit a linear model to this dataset.": {
            "org.modellab.statistics.dataset",
            "org.modellab.statistics.linear-model-study",
        },
    }
    for instruction, expected in cases.items():
        assert set(_official_kind_context_selection(instruction, None)) == expected


def test_router_does_not_promote_property_homonyms_to_unrelated_packs() -> None:
    from model_lab.interpreter import _official_kind_context_selection

    assert set(_official_kind_context_selection(
        "Create a Markov chain with this transition matrix.", None
    )) == {"org.modellab.probability.markov-chain"}
    assert set(_official_kind_context_selection(
        "Build an HMM with a transition matrix.", None
    )) == {"org.modellab.generative.hidden-markov-model"}
    assert _official_kind_context_selection(
        "Use inverse temperature beta in a probability model.", None
    ) == ()


def test_entire_current_corpus_has_no_exact_identifier_boilerplate() -> None:
    records = _records("train") + _records("validation")
    assert len(records) == 6300
    assert all(
        "use these exact identifiers" not in record["conversation"]["instruction"].casefold()
        for record in records
    )


def test_router_respects_clause_scoped_negation() -> None:
    mechanics = "org.modellab.pack.mechanics-structures-materials"
    optimisation = "org.modellab.pack.optimisation-estimation-inverse-problems"
    graph = "org.modellab.pack.graphs-networks-discrete"
    for instruction in (
        "Without changing the topology, optimise the truss.",
        "Analyse the truss without changing it, then optimise it.",
        "Without adding a graph, optimise the truss.",
        "Do not use a graph representation; optimise the truss.",
        "Avoid changing the graph; optimise the truss.",
    ):
        packs = set(_official_pack_context_selection(instruction, None))
        assert mechanics in packs
        assert optimisation in packs
        assert graph not in packs


def test_router_treats_scientific_matrices_as_object_properties() -> None:
    from model_lab.interpreter import _official_kind_context_selection

    cases = {
        "Create a population model with an interaction matrix.": {
            "org.modellab.biological.population-interaction-system",
        },
        "Create a feature matrix for binary classification.": {
            "org.modellab.learning.feature-dataset",
            "org.modellab.learning.supervised-study",
        },
        "Fit a linear regression with a design matrix.": {
            "org.modellab.statistics.linear-model-study",
        },
    }
    for instruction, expected in cases.items():
        assert set(_official_kind_context_selection(instruction, None)) == expected


def test_router_disambiguates_species_and_objective_homonyms() -> None:
    from model_lab.interpreter import _official_kind_context_selection

    assert set(_official_kind_context_selection(
        "Create an ecological model with three species.", None
    )) == {"org.modellab.biological.population-interaction-system"}
    assert set(_official_kind_context_selection(
        "Create an objective measurement with standard uncertainty.", None
    )) == {"org.modellab.multidimensional.quantity"}


def test_current_corpus_model_naming_is_diversified() -> None:
    records = _records("train") + _records("validation")
    instructions = [record["conversation"]["instruction"].casefold() for record in records]
    assert all("call the model" not in text for text in instructions)
    assert all("name the model" not in text for text in instructions)
    naming_forms = (
        "model should be titled",
        "give this model the name",
        "save the model as",
        "let the model name be",
        "title the resulting model",
        "requested model is named",
        "label the model",
    )
    assert max(sum(form in text for text in instructions) for form in naming_forms) < 600


def test_router_handles_alternative_exclusions_and_while_scope() -> None:
    mechanics = "org.modellab.pack.mechanics-structures-materials"
    optimisation = "org.modellab.pack.optimisation-estimation-inverse-problems"
    graph = "org.modellab.pack.graphs-networks-discrete"
    learning = "org.modellab.pack.machine-learning-computational-intelligence"

    assert set(_official_pack_context_selection(
        "Use a neural network instead of a graph.", None
    )) == {learning}
    assert set(_official_pack_context_selection(
        "Use a neural network rather than a graph.", None
    )) == {learning}
    assert set(_official_pack_context_selection(
        "Use a graph instead of a neural network.", None
    )) == {graph}
    for instruction in (
        "Optimise the truss without ever creating a graph.",
        "Instead of creating a graph, optimise the truss.",
        "Rather than creating a graph, optimise the truss.",
        "Do not add a graph while optimising the truss.",
    ):
        assert set(_official_pack_context_selection(instruction, None)) == {mechanics, optimisation}


def test_router_distinguishes_plotting_graph_from_graph_theory() -> None:
    for instruction in (
        "Graph y = sin(x).",
        "Graph this function.",
        "Plot the graph of y=x^2.",
    ):
        assert "org.modellab.pack.graphs-networks-discrete" not in set(
            _official_pack_context_selection(instruction, None)
        )
        assert "org.modellab.graph.network" not in set(
            _official_kind_context_selection(instruction, None)
        )


def test_router_distinguishes_statistical_population_from_biology() -> None:
    biological = "org.modellab.pack.chemical-reaction-biological-systems"
    statistics = "org.modellab.pack.statistical-inference-data-modelling"
    estimation = "org.modellab.pack.optimisation-estimation-inverse-problems"

    packs = set(_official_pack_context_selection(
        "Estimate the population mean from this dataset.", None
    ))
    assert statistics in packs and estimation in packs and biological not in packs
    packs = set(_official_pack_context_selection(
        "Create a statistical model for a population sample.", None
    ))
    assert statistics in packs and biological not in packs


def test_router_treats_additional_domain_matrices_as_properties() -> None:
    array_kind = "org.modellab.multidimensional.array"
    cases = {
        "Fit a statistical model with a covariance matrix.",
        "Create a classifier and report its confusion matrix.",
        "Analyse a truss using its stiffness matrix.",
        "Solve this ODE using a Jacobian matrix.",
        "Create a graph represented by an adjacency matrix.",
    }
    for instruction in cases:
        assert array_kind not in set(_official_kind_context_selection(instruction, None))
        assert "org.modellab.pack.multidimensional-mathematics" not in set(
            _official_pack_context_selection(instruction, None)
        )


def test_router_handles_coordinated_exclusions_and_representation_negation() -> None:
    dynamics = "org.modellab.pack.dynamics-differential-equations-control"
    mechanics = "org.modellab.pack.mechanics-structures-materials"
    optimisation = "org.modellab.pack.optimisation-estimation-inverse-problems"
    for instruction in (
        "Use an ODE, not a graph or neural network.",
        "Create an ODE instead of a graph and a neural network.",
        "Neither create a graph nor a neural network; build an ODE.",
    ):
        assert set(_official_pack_context_selection(instruction, None)) == {dynamics}
    assert set(_official_pack_context_selection(
        "Optimise the truss, but not with a graph representation.", None
    )) == {mechanics, optimisation}


def test_router_distinguishes_general_plotting_from_network_graphing() -> None:
    graph_pack = "org.modellab.pack.graphs-networks-discrete"
    graph_kind = "org.modellab.graph.network"
    for instruction in (
        "Graph the trajectory over time.",
        "Graph the phase portrait.",
        "Graph the residuals.",
        "Graph the results.",
    ):
        assert graph_pack not in set(_official_pack_context_selection(instruction, None))
        assert graph_kind not in set(_official_kind_context_selection(instruction, None))
    assert set(_official_kind_context_selection(
        "Plot a graph of this network.", None
    )) == {graph_kind}


def test_router_handles_wider_statistical_population_language() -> None:
    biological = "org.modellab.pack.chemical-reaction-biological-systems"
    for instruction in (
        "Estimate the population proportion.",
        "Use the population standard deviation.",
        "Report the population quantile.",
        "Estimate a population percentile.",
        "Compute the population median.",
    ):
        assert biological not in set(_official_pack_context_selection(instruction, None))
        assert "org.modellab.biological.population-interaction-system" not in set(
            _official_kind_context_selection(instruction, None)
        )


def test_router_generalises_matrix_property_suppression() -> None:
    array_kind = "org.modellab.multidimensional.array"
    array_pack = "org.modellab.pack.multidimensional-mathematics"
    cases = (
        "Use a correlation matrix for PCA.",
        "Use a Hessian matrix for this optimisation.",
        "Use a kernel matrix for the classifier.",
        "Use a Fisher information matrix for parameter estimation.",
        "Use a sensitivity matrix for the ODE.",
        "Use an incidence matrix for the graph.",
        "Use a Laplacian matrix for the graph.",
    )
    for instruction in cases:
        assert array_kind not in set(_official_kind_context_selection(instruction, None))
        assert array_pack not in set(_official_pack_context_selection(instruction, None))
    assert "org.modellab.pde.poisson-problem" not in set(
        _official_kind_context_selection("Use a Laplacian matrix for the graph.", None)
    )


def test_router_covers_common_network_domains_and_path_language() -> None:
    graph_kind = "org.modellab.graph.network"
    for instruction in (
        "Create a social network.",
        "Create a citation network.",
        "Create a transport network.",
        "Find the shortest path in a network.",
    ):
        assert set(_official_kind_context_selection(instruction, None)) == {graph_kind}


def test_current_corpus_avoids_dominant_model_naming_constructions() -> None:
    records = _records("train") + _records("validation")
    instructions = [record["conversation"]["instruction"].casefold() for record in records]
    legacy_forms = (
        "in a model named",
        "model should be titled",
        "give this model the name",
        "save the model as",
        "let the model name be",
        "title the resulting model",
        "requested model is named",
        "label the model",
    )
    combined = sum(any(form in text for form in legacy_forms) for text in instructions)
    assert combined < len(instructions) * 0.20
    for form in legacy_forms:
        assert sum(form in text for text in instructions) < len(instructions) * 0.05


def test_rich_model_property_edits_infer_unique_existing_kind() -> None:
    source = (ROOT / "models" / "generative-systems.yaml").read_text(encoding="utf-8")
    cases = {
        "Update the emission matrix.": "org.modellab.generative.hidden-markov-model",
        "Set the policy precision to 4.": "org.modellab.generative.active-inference-model",
        "Change the rewards.": "org.modellab.generative.pomdp",
        "Set discount to 0.9.": "org.modellab.generative.pomdp",
        "Update the likelihood.": "org.modellab.generative.active-inference-model",
    }
    for instruction, expected in cases.items():
        context = create_interpreter_context(
            instruction=instruction,
            current_model_source=source,
        )
        official = {
            item["kind"] for item in context["contract"]["installed_model_object_kinds"]
            if not item["kind"].startswith("org.modellab.core.")
        }
        assert official == {expected}, (instruction, official)


def test_router_handles_except_for_and_as_opposed_to_exclusions() -> None:
    dynamics = "org.modellab.pack.dynamics-differential-equations-control"
    graph = "org.modellab.pack.graphs-networks-discrete"
    for instruction in (
        "Use an ODE except for a graph.",
        "Use an ODE as opposed to a graph.",
    ):
        packs = set(_official_pack_context_selection(instruction, None))
        assert dynamics in packs and graph not in packs


def test_router_treats_draw_and_sketch_graph_as_plotting() -> None:
    graph_pack = "org.modellab.pack.graphs-networks-discrete"
    graph_kind = "org.modellab.graph.network"
    for instruction in (
        "Draw the graph of y = sin(x).",
        "Draw a graph of this function.",
        "Sketch the graph of y=x^2.",
    ):
        assert graph_pack not in set(_official_pack_context_selection(instruction, None))
        assert graph_kind not in set(_official_kind_context_selection(instruction, None))


def test_router_treats_graph_laplacian_as_graph_property() -> None:
    graph_kind = "org.modellab.graph.network"
    pde_kind = "org.modellab.pde.poisson-problem"
    pde_pack = "org.modellab.pack.spatial-fields-continuum-pdes"
    for instruction in (
        "Compute graph Laplacian eigenvalues.",
        "Compute the Laplacian of the network.",
        "Use the network Laplacian for spectral clustering.",
    ):
        kinds = set(_official_kind_context_selection(instruction, None))
        assert graph_kind in kinds and pde_kind not in kinds
        assert pde_pack not in set(_official_pack_context_selection(instruction, None))


def test_router_covers_road_network_language() -> None:
    assert set(_official_kind_context_selection("Create a road network.", None)) == {
        "org.modellab.graph.network"
    }


def test_current_corpus_avoids_repeated_grounding_and_closing_boilerplate() -> None:
    records = _records("train") + _records("validation")
    instructions = [record["conversation"]["instruction"] for record in records]
    assert sum('Use "choice" as the ambiguity name.' in text for text in instructions) < 50
    repeated_closings = (
        "Those are the specifications for the model.",
        "Treat those details as the model specification.",
        "Please represent that in Model Laboratory.",
        "Please turn that into a Model Laboratory model.",
        "Please build the model from those requirements.",
        "That description should define the model.",
        "Please construct the model described above.",
        "Represent those requirements in the model.",
        "Use the preceding description for the model.",
        "Please capture those requirements in the model.",
        "Use those details for the resulting model.",
        "That is the model I want to construct.",
        "Please model the system described above.",
    )
    for sentence in repeated_closings:
        assert sum(sentence in text for text in instructions) < 100, sentence


def test_unsupported_and_unimplemented_compositions_have_a_fail_closed_boundary() -> None:
    unsupported = (
        "Create a stochastic differential equation.",
        "Train a neural network on these observations.",
        "Create a graph neural network.",
        "Build a neural controller for an ODE system.",
        "Optimize a truss for minimum mass.",
        "Estimate the parameters of this ODE.",
        "Fit an RLC circuit to measurements.",
        "Solve Laplace's equation on a triangle mesh.",
    )
    for instruction in unsupported:
        reason = scientific_request_boundary(instruction)
        assert reason
        context = create_interpreter_context(
            instruction=instruction, current_model_source=""
        )
        assert context["contract"]["request_boundary"] == {
            "status": "unsupported-by-installed-capabilities",
            "required_action": "unable",
            "reason": reason,
        }


def test_supported_neighbours_do_not_trigger_the_composition_boundary() -> None:
    for instruction in (
        "Analyse the truss without optimising it.",
        "Create a fixed feed-forward network with explicit weights.",
        "Create a deterministic ordinary differential equation.",
        "Solve a Poisson equation on a uniform rectilinear grid.",
        "Fit an explicit nonlinear residual system to data.",
        "Create a mass-action reaction network identified as reactor with rate constants 0.2 and 0.4.",
    ):
        assert scientific_request_boundary(instruction) is None, instruction


def test_compiler_rejects_non_unable_output_for_a_boundary_request() -> None:
    instruction = "Estimate the parameters of this ODE."
    context = create_interpreter_context(
        instruction=instruction, current_model_source=""
    )
    output = {
        "schema": INTERPRETER_OUTPUT_SCHEMA,
        "schema_version": INTERPRETER_OUTPUT_SCHEMA_VERSION,
        "action": "needs_clarification",
        "context_sha256": context["context_sha256"],
        "context_requests": [],
        "operations": [],
        "explanation": "",
        "warnings": [],
        "clarification_question": "Which observations should be used?",
    }
    with pytest.raises(InterpreterProposalError, match="requires an unable response"):
        process_interpreter_output(
            raw_output=output,
            context_document=context,
            provider_identity=None,
            instruction=instruction,
            current_model_source="",
        )
