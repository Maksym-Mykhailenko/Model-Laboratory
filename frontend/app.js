import {
  base64ToText,
  defaultExperimentName,
  escapeHtml,
  metricHtml,
  reportTone,
  resultInvalidationForInput,
  liveCommitRelationship,
  scientificDisplay,
  shouldCommitResult,
  shortHash,
  statusTone,
  tableHtml,
  textToBase64,
  validateNumericalSettings,
} from "./core.mjs";

const byId = (id) => document.getElementById(id);
const invoke = window.__TAURI__?.core?.invoke;

const elements = {
  source: byId("model-source"),
  documentTitle: byId("document-title"),
  validationChip: byId("validation-chip"),
  sourcePosition: byId("source-position"),
  statusMessage: byId("status-message"),
  statusContext: byId("status-context"),
  engineDot: byId("engine-dot"),
  engineLabel: byId("engine-label"),
  busyLayer: byId("busy-layer"),
  busyTitle: byId("busy-title"),
  busyDetail: byId("busy-detail"),
  toastRegion: byId("toast-region"),
  modelPlot: byId("model-plot"),
  plotEmpty: byId("plot-empty"),
  visualisationSelect: byId("visualisation-select"),
  parameterControls: byId("parameter-controls"),
  capabilityList: byId("capability-list"),
  officialPackCatalogue: byId("official-pack-catalogue"),
  capabilitySelect: byId("capability-select"),
  capabilityDescription: byId("capability-description"),
  capabilitySettings: byId("capability-settings"),
  capabilityStatus: byId("capability-status"),
  capabilityPlan: byId("capability-plan"),
  capabilityResult: byId("capability-result"),
  capabilityPlot: byId("capability-plot"),
  workloadContent: byId("workload-content"),
  structureContent: byId("structure-content"),
  symbolicContent: byId("symbolic-content"),
  stationaryContent: byId("stationary-content"),
  stationaryStatus: byId("stationary-status"),
  provenanceContent: byId("provenance-content"),
  evaluationDiagnostics: byId("evaluation-diagnostics"),
  sweepForm: byId("sweep-form"),
  sweepParameter: byId("sweep-parameter"),
  sweepStart: byId("sweep-start"),
  sweepEnd: byId("sweep-end"),
  sweepSteps: byId("sweep-steps"),
  sweepVisualisation: byId("sweep-visualisation"),
  sweepUnavailable: byId("sweep-unavailable"),
  sweepResults: byId("sweep-results"),
  sweepMetrics: byId("sweep-metrics"),
  sweepPlot: byId("sweep-plot"),
  sweepSummary: byId("sweep-summary"),
  openedExperiment: byId("opened-experiment"),
  experimentStatus: byId("experiment-status"),
  liveStateStatus: byId("live-state-status"),
  committedStateStatus: byId("committed-state-status"),
  committedStateHash: byId("committed-state-hash"),
  committedExperimentHash: byId("committed-experiment-hash"),
  committedAt: byId("committed-at"),
  committedAuthority: byId("committed-authority"),
  authorReview: byId("author-review"),
  reviewHash: byId("review-hash"),
  reviewSummary: byId("review-summary"),
  reviewJson: byId("review-json"),
  approveReview: byId("approve-review"),
  reproductionContent: byId("reproduction-content"),
  interpreterStatus: byId("interpreter-status"),
  interpreterRuntime: byId("interpreter-runtime"),
  interpreterModel: byId("interpreter-model"),
  interpreterModelDigest: byId("interpreter-model-digest"),
  interpreterCandidateLifecycle: byId("interpreter-candidate-lifecycle"),
  interpreterTrainingReport: byId("interpreter-training-report"),
  interpreterEvaluationReport: byId("interpreter-evaluation-report"),
  interpreterRuntimeMessage: byId("interpreter-runtime-message"),
  interpreterInstruction: byId("interpreter-instruction"),
  interpreterAttachButton: byId("interpreter-attach-button"),
  interpreterAttachmentList: byId("interpreter-attachment-list"),
  interpreterClarification: byId("interpreter-clarification"),
  interpreterClarificationQuestion: byId("interpreter-clarification-question"),
  interpreterClarificationHistory: byId("interpreter-clarification-history"),
  interpreterClarificationAnswer: byId("interpreter-clarification-answer"),
  interpreterProposal: byId("interpreter-proposal"),
  interpreterProposalHash: byId("interpreter-proposal-hash"),
  interpreterProposalMetrics: byId("interpreter-proposal-metrics"),
  interpreterExplanation: byId("interpreter-explanation"),
  interpreterWarningsSection: byId("interpreter-warnings-section"),
  interpreterWarnings: byId("interpreter-warnings"),
  interpreterDiff: byId("interpreter-diff"),
  interpreterModelSummary: byId("interpreter-model-summary"),
  approveInterpreterProposal: byId("approve-interpreter-proposal"),
};

const state = {
  source: "",
  sourceName: "Untitled model",
  sourcePath: null,
  model: null,
  analysis: null,
  sweep: null,
  capabilityResult: null,
  capabilityPlan: [],
  openedDocument: null,
  experimentInspection: null,
  reproduction: null,
  prepared: null,
  committed: null,
  committedInputRevision: null,
  busyCount: 0,
  requestCounter: 0,
  inputRevision: 0,
  interpreterRuntime: null,
  interpreterProposal: null,
  interpreterBusy: false,
  interpreterAcceptances: [],
  interpreterConversation: null,
  interpreterClarificationHistory: [],
  pendingInterpreterQuestion: null,
  interpreterAttachments: [],
};

function setStatus(message, context = null) {
  elements.statusMessage.textContent = message;
  if (context !== null) elements.statusContext.textContent = context;
}

function setChip(element, text, tone = "neutral") {
  element.textContent = text;
  element.className = `status-chip ${tone}`;
}

function renderCommittedBoundary() {
  const relationship = liveCommitRelationship(state.committedInputRevision, state.inputRevision);
  if (!state.committed) {
    setChip(elements.liveStateStatus, "Live · uncommitted", "warning");
    setChip(elements.committedStateStatus, "No commit", "neutral");
    elements.committedStateHash.textContent = "—";
    elements.committedExperimentHash.textContent = "—";
    elements.committedAt.textContent = "—";
    elements.committedAuthority.textContent = "None — live state is never external-execution authority";
    byId("save-commit-button").disabled = true;
  } else {
    const matches = relationship === "matches";
    setChip(
      elements.liveStateStatus,
      matches ? "Live · matches commit" : "Live · diverged",
      matches ? "success" : "warning",
    );
    setChip(elements.committedStateStatus, "Committed", "success");
    elements.committedStateHash.textContent = state.committed.commit_sha256;
    elements.committedStateHash.title = state.committed.commit_sha256;
    elements.committedExperimentHash.textContent = state.committed.experiment_state_sha256;
    elements.committedExperimentHash.title = state.committed.experiment_state_sha256;
    elements.committedAt.textContent = state.committed.committed_at_utc;
    elements.committedAuthority.textContent = matches
      ? "Committed snapshot is stable; live values currently derive from that commit"
      : "Committed snapshot remains authoritative for external execution; live edits are isolated";
    byId("save-commit-button").disabled = false;
  }
  byId("commit-state-button").disabled = !state.prepared;
}

function toast(message, tone = "normal") {
  const item = document.createElement("div");
  item.className = `toast ${tone === "error" ? "error" : ""}`;
  item.textContent = message;
  elements.toastRegion.append(item);
  window.setTimeout(() => item.remove(), tone === "error" ? 8000 : 4500);
}

async function native(command, payload = {}) {
  if (!invoke) throw new Error("The native Tauri bridge is unavailable. Start the app with ‘npm run desktop:dev’. ");
  return invoke(command, payload);
}

async function engine(action, payload = {}) {
  state.requestCounter += 1;
  const request = JSON.stringify({ id: state.requestCounter, action, payload });
  const raw = await native("engine_request", { requestJson: request });
  const response = JSON.parse(raw);
  if (!response.ok) throw new Error(response.error?.message || "The scientific engine rejected the request.");
  return response.result;
}

async function withBusy(title, detail, operation) {
  state.busyCount += 1;
  elements.busyTitle.textContent = title;
  elements.busyDetail.textContent = detail;
  elements.busyLayer.classList.remove("hidden");
  try {
    return await operation();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    toast(message, "error");
    throw error;
  } finally {
    state.busyCount -= 1;
    if (state.busyCount <= 0) {
      state.busyCount = 0;
      elements.busyLayer.classList.add("hidden");
    }
  }
}

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  });
  if (name === "visualisation" && state.analysis?.figure) resizePlot(elements.modelPlot);
  if (name === "capabilities" && state.capabilityResult?.figure) resizePlot(elements.capabilityPlot);
  if (name === "sweep" && state.sweep?.figure) resizePlot(elements.sweepPlot);
}

function updateSourcePosition() {
  const lines = elements.source.value ? elements.source.value.split(/\r?\n/).length : 0;
  elements.sourcePosition.textContent = `${lines} ${lines === 1 ? "line" : "lines"}`;
}

function clearPreparedState(reason = null) {
  if (state.prepared && reason) toast(reason);
  state.prepared = null;
  elements.authorReview.classList.add("hidden");
  elements.approveReview.checked = false;
  byId("save-draft-button").disabled = true;
  byId("save-state-button").disabled = true;
  byId("publish-experiment-button").disabled = true;
  setChip(elements.experimentStatus, "Draft not prepared", "neutral");
  renderCommittedBoundary();
}

function clearInterpreterProposal() {
  state.interpreterProposal = null;
  elements.interpreterProposal.classList.add("hidden");
  elements.approveInterpreterProposal.checked = false;
  byId("interpreter-accept-button").disabled = true;
  elements.interpreterProposalHash.textContent = "";
  elements.interpreterProposalMetrics.replaceChildren();
  elements.interpreterWarnings.replaceChildren();
  elements.interpreterDiff.textContent = "";
  elements.interpreterModelSummary.replaceChildren();
}

function clearInterpreterClarification(resetHistory = false) {
  state.pendingInterpreterQuestion = null;
  elements.interpreterClarification.classList.add("hidden");
  elements.interpreterClarificationQuestion.textContent = "";
  elements.interpreterClarificationAnswer.value = "";
  elements.interpreterClarificationHistory.replaceChildren();
  if (resetHistory) {
    state.interpreterConversation = null;
    state.interpreterClarificationHistory = [];
  }
}

function renderInterpreterClarification(question) {
  state.pendingInterpreterQuestion = question;
  elements.interpreterClarificationQuestion.textContent = question;
  elements.interpreterClarificationHistory.innerHTML = state.interpreterClarificationHistory
    .map((turn, index) => `<div class="clarification-turn"><strong>Earlier answer ${index + 1}</strong>${escapeHtml(turn.question)}<br>${escapeHtml(turn.answer)}</div>`)
    .join("");
  elements.interpreterClarificationAnswer.value = "";
  elements.interpreterClarification.classList.remove("hidden");
  elements.interpreterClarificationAnswer.focus();
  elements.interpreterClarification.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearInterpreterLineage() {
  state.interpreterAcceptances = [];
}

function interpreterAttachmentIds() {
  return state.interpreterAttachments
    .map((item) => item.attachment_id)
    .sort();
}

function sameStringArray(left, right) {
  return left.length === right.length && left.every((item, index) => item === right[index]);
}

function renderInterpreterAttachments() {
  elements.interpreterAttachmentList.replaceChildren();
  if (!state.interpreterAttachments.length) {
    const empty = document.createElement("p");
    empty.className = "interpreter-attachment-empty";
    empty.textContent = "No reference files attached.";
    elements.interpreterAttachmentList.append(empty);
    return;
  }
  state.interpreterAttachments.forEach((attachment) => {
    const row = document.createElement("div");
    row.className = "interpreter-attachment-item";
    const description = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = attachment.name;
    const evidence = document.createElement("span");
    evidence.textContent = `${attachment.media_type} · ${(attachment.source_size_bytes / 1024).toFixed(1)} KiB · ${shortHash(attachment.source_sha256)}`;
    description.append(name, evidence);
    const remove = document.createElement("button");
    remove.className = "text-button";
    remove.type = "button";
    remove.textContent = "Remove";
    remove.disabled = state.interpreterBusy;
    remove.addEventListener("click", () => removeInterpreterAttachment(attachment.attachment_id).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(message);
      toast(message, "error");
    }));
    row.append(description, remove);
    elements.interpreterAttachmentList.append(row);
  });
}

function invalidateInterpreterConversationForAttachments() {
  clearInterpreterProposal();
  clearInterpreterClarification(true);
  syncInterpreterControls();
}

async function attachInterpreterFile() {
  if (state.interpreterAttachments.length >= 8) {
    throw new Error("At most eight reference files may be attached at once.");
  }
  const documentFile = await native("open_document", { kind: "interpreter-attachment" });
  if (!documentFile) return;
  setInterpreterBusy(true, "local attachment extraction");
  try {
    const result = await engine("ingest_interpreter_attachment", {
      name: documentFile.name,
      data_base64: documentFile.dataBase64,
    });
    const attachment = result.attachment;
    if (!state.interpreterAttachments.some((item) => item.attachment_id === attachment.attachment_id)) {
      state.interpreterAttachments.push(attachment);
      state.interpreterAttachments.sort((left, right) => left.attachment_id.localeCompare(right.attachment_id));
    }
    invalidateInterpreterConversationForAttachments();
    renderInterpreterAttachments();
    setStatus("Reference file attached locally.", `${attachment.name} · ${shortHash(attachment.source_sha256)}`);
  } finally {
    setInterpreterBusy(false);
  }
}

async function removeInterpreterAttachment(attachmentId) {
  setInterpreterBusy(true, "attachment removal");
  try {
    await engine("release_interpreter_attachment", { attachment_id: attachmentId });
    state.interpreterAttachments = state.interpreterAttachments.filter(
      (item) => item.attachment_id !== attachmentId,
    );
    invalidateInterpreterConversationForAttachments();
    renderInterpreterAttachments();
    setStatus("Reference file removed. Any pending interpreter dialogue was cleared.");
  } finally {
    setInterpreterBusy(false);
  }
}

function purgePlot(target) {
  if (window.Plotly && target?.data && typeof window.Plotly.purge === "function") {
    window.Plotly.purge(target);
  } else {
    target?.replaceChildren();
  }
}

function clearSweepView() {
  state.sweep = null;
  purgePlot(elements.sweepPlot);
  elements.sweepMetrics.replaceChildren();
  elements.sweepSummary.replaceChildren();
  elements.sweepResults.classList.add("hidden");
}

function clearCapabilityResult() {
  state.capabilityResult = null;
  purgePlot(elements.capabilityPlot);
  elements.capabilityPlot.classList.add("hidden");
  elements.capabilityResult.className = "content-stack empty-content";
  elements.capabilityResult.textContent = "No capability has been run.";
  setChip(elements.capabilityStatus, "No run", "neutral");
}

function clearCapabilityPlan() {
  state.capabilityPlan = [];
  renderCapabilityPlan();
}

function clearReproductionView() {
  state.reproduction = null;
  elements.reproductionContent.className = "empty-state compact-empty";
  elements.reproductionContent.innerHTML = `
    <div class="empty-glyph" aria-hidden="true">✓</div>
    <h3>No current reproduction report</h3>
    <p>Open an experiment, inspect its workload, then explicitly reproduce it.</p>
  `;
}

function invalidateComputedResults(identifier, reason) {
  const scope = resultInvalidationForInput(identifier);
  state.inputRevision += 1;
  clearPreparedState(reason);
  if (scope.analysis) {
    clearAnalysisViews();
  } else if (scope.sweep) {
    clearSweepView();
  }
  if (scope.reproduction) clearReproductionView();
  clearCapabilityResult();
  if (state.capabilityPlan.length) clearCapabilityPlan();
  setStatus("Experiment inputs changed. Run the affected analysis again.");
  renderCommittedBoundary();
}

function invalidateModel() {
  state.inputRevision += 1;
  state.model = null;
  state.analysis = null;
  state.sweep = null;
  clearPreparedState();
  elements.source.classList.remove("invalid");
  setChip(elements.validationChip, "Changed", "warning");
  byId("analyse-button").disabled = true;
  byId("prepare-experiment-button").disabled = true;
  byId("reset-parameters-button").disabled = true;
  byId("run-sweep-button").disabled = true;
  clearAnalysisViews();
  clearCapabilityResult();
  clearCapabilityPlan();
  clearReproductionView();
  renderCommittedBoundary();
}

function setSource(source, name = "Untitled model", path = null) {
  clearInterpreterProposal();
  clearInterpreterClarification(true);
  clearInterpreterLineage();
  state.source = source;
  state.sourceName = name;
  state.sourcePath = path;
  elements.source.value = source;
  elements.documentTitle.textContent = name;
  updateSourcePosition();
  invalidateModel();
  setChip(elements.validationChip, "Not validated", "neutral");
}

function clearAnalysisViews() {
  state.analysis = null;
  elements.plotEmpty.classList.remove("hidden");
  elements.modelPlot.classList.add("hidden");
  purgePlot(elements.modelPlot);
  elements.evaluationDiagnostics.replaceChildren();
  elements.symbolicContent.className = "formula-stack empty-content";
  elements.symbolicContent.textContent = "Run an analysis to derive symbolic results.";
  elements.stationaryContent.className = "content-stack empty-content";
  elements.stationaryContent.textContent = "Run an analysis to locate stationary points.";
  setChip(elements.stationaryStatus, "Not analysed", "neutral");
  elements.provenanceContent.className = "content-stack empty-content";
  elements.provenanceContent.textContent = "Run an analysis to construct the provenance record.";
  elements.workloadContent.className = "empty-setting";
  elements.workloadContent.textContent = "Calculated before execution.";
  clearSweepView();
  clearCapabilityResult();
}

function applyDefaults(model) {
  const defaults = model.defaults || {};
  const evaluation = defaults.evaluation_settings || {};
  const stationary = defaults.stationary_settings || {};
  const reproduction = defaults.reproduction_settings || {};
  byId("points-1d").value = evaluation.points_1d ?? 1000;
  byId("points-2d").value = evaluation.points_per_axis_2d ?? 150;
  byId("samples-1d").value = stationary.samples_1d ?? 2001;
  byId("seeds-2d").value = stationary.seeds_per_axis_2d ?? 11;
  byId("root-tolerance").value = stationary.root_tolerance ?? 1e-9;
  byId("relative-tolerance").value = reproduction.relative_tolerance ?? 1e-8;
  byId("absolute-tolerance").value = reproduction.absolute_tolerance ?? 1e-11;
}

function renderModelIdentity(model) {
  const rows = byId("model-identity").querySelectorAll("dd");
  rows[0].textContent = model.name;
  rows[1].textContent = model.counts.variables;
  rows[2].textContent = model.counts.parameters;
  rows[3].textContent = shortHash(model.source_sha256, 9, 6);
  rows[3].title = model.source_sha256;
}

function renderCapabilities(model) {
  const capabilities = model.capabilities;
  const items = [
    ...capabilities.visualisations.map((value) => [value, ""]),
    ...capabilities.analyses.map((value) => [value, "analysis"]),
    ...capabilities.controls.map((value) => [value, "muted"]),
    ...capabilities.installed_packs
      .filter((item) => item.applicable)
      .map((item) => [item.title, "analysis"]),
  ];
  elements.capabilityList.innerHTML = items.length
    ? items.map(([label, kind]) => `<span class="tag ${kind}">${escapeHtml(label)}</span>`).join("")
    : '<span class="tag muted">Validation only</span>';
  const packs = capabilities.official_packs || [];
  elements.officialPackCatalogue.className = packs.length ? "pack-catalogue" : "pack-catalogue empty-content";
  elements.officialPackCatalogue.innerHTML = packs.length
    ? packs.map((pack) => `
      <article class="pack-card">
        <div><span class="status-chip ${pack.installed ? "success" : "neutral"}">${pack.installed ? "Installed" : "Available"}</span><span class="pack-version">v${escapeHtml(pack.version)}</span></div>
        <h4>${escapeHtml(pack.title)}</h4>
        <p>${escapeHtml(pack.description)}</p>
        <small>${pack.applicable_capability_count} of ${pack.capability_count} capabilities apply to this model · ${pack.object_kinds.length} object kinds</small>
      </article>`).join("")
    : "No official pack metadata is available.";
}

function selectedCapability() {
  return state.model?.capabilities.installed_packs.find(
    (item) => item.id === elements.capabilitySelect.value,
  ) || null;
}

function defaultCapabilitySettings(capability, model) {
  const settings = {};
  const properties = capability?.settings_schema?.properties || {};
  for (const [name, specification] of Object.entries(properties)) {
    if (Object.hasOwn(specification, "default")) settings[name] = specification.default;
  }
  const parameters = Object.fromEntries(
    model.structure.parameters.map((item) => [item.name, item.default]),
  );
  const midpoint = model.structure.variables.map((item) => (item.lower + item.upper) / 2);
  if (Object.hasOwn(properties, "parameter_values")) settings.parameter_values = parameters;
  if (Object.hasOwn(properties, "function_name")) {
    const collection = capability.id.startsWith("org.modellab.vector.")
      ? model.structure.vector_functions
      : capability.id.startsWith("org.modellab.matrix.")
        ? model.structure.matrix_functions
        : model.structure.functions;
    if (collection.length > 1) settings.function_name = collection[0].name;
  }
  if (capability.id === "org.modellab.scalar.evaluate-points") settings.points = [midpoint];
  if (capability.id === "org.modellab.scalar.evaluate-curve") settings.points = 1000;
  if (capability.id === "org.modellab.scalar.evaluate-surface") settings.points_per_axis = 120;
  if (capability.id === "org.modellab.scalar.stationary-points") Object.assign(settings, { samples_1d: 2001, seeds_2d: 11 });
  if (capability.id === "org.modellab.vector.solve-roots") settings.seeds = 64;
  if (capability.id === "org.modellab.vector.field-2d") Object.assign(settings, { points_per_axis: 25, root_seeds: 64 });
  if (capability.id === "org.modellab.matrix.analyse") settings.point = midpoint;
  if (capability.id === "org.modellab.optimization.constrained") Object.assign(settings, { objective: "minimize", seeds: 32 });
  const graphObjects = model.structure.model_graph?.objects || [];
  const kindByCapability = {
    "org.modellab.multidimensional.analyse-array": "org.modellab.multidimensional.array",
    "org.modellab.probability.analyse-distribution": "org.modellab.probability.discrete-distribution",
    "org.modellab.probability.evolve-markov-chain": "org.modellab.probability.markov-chain",
    "org.modellab.probability.simulate-markov-chain": "org.modellab.probability.markov-chain",
    "org.modellab.graph.analyse-network": "org.modellab.graph.network",
    "org.modellab.generative.infer-hidden-markov-model": "org.modellab.generative.hidden-markov-model",
    "org.modellab.generative.solve-pomdp": "org.modellab.generative.pomdp",
    "org.modellab.generative.evaluate-active-inference": "org.modellab.generative.active-inference-model",
    "org.modellab.dynamics.integrate-ode": "org.modellab.dynamics.ode-system",
    "org.modellab.dynamics.analyse-equilibria": "org.modellab.dynamics.ode-system",
    "org.modellab.control.analyse-state-space": "org.modellab.control.state-space-system",
    "org.modellab.control.simulate-state-space": "org.modellab.control.state-space-system",
    "org.modellab.field.analyse-scalar-field": "org.modellab.field.structured-scalar-field",
    "org.modellab.field.analyse-vector-field": "org.modellab.field.structured-vector-field",
    "org.modellab.pde.solve-diffusion": "org.modellab.pde.diffusion-problem",
    "org.modellab.pde.solve-poisson": "org.modellab.pde.poisson-problem",
    "org.modellab.geometry.analyse-point-cloud": "org.modellab.geometry.point-cloud",
    "org.modellab.geometry.analyse-triangle-mesh": "org.modellab.geometry.triangle-mesh",
    "org.modellab.geometry.shortest-mesh-path": "org.modellab.geometry.triangle-mesh",
    "org.modellab.material.analyse-isotropic-elasticity": "org.modellab.material.isotropic-linear-elastic",
    "org.modellab.mechanics.solve-truss-static": "org.modellab.mechanics.truss-structure",
    "org.modellab.mechanics.analyse-truss-modes": "org.modellab.mechanics.truss-structure",
  };
  const selectedObject = graphObjects.find((item) => item.kind === kindByCapability[capability.id]);
  if (selectedObject && Object.hasOwn(properties, "object_id")) settings.object_id = selectedObject.id;
  if (capability.id === "org.modellab.generative.infer-hidden-markov-model" && selectedObject) {
    settings.observed_sequence = [selectedObject.properties.observations[0]];
  }
  if (capability.id === "org.modellab.generative.markov-blanket") {
    const network = graphObjects.find((item) => item.kind === "org.modellab.graph.network" && item.properties.directed);
    if (network) Object.assign(settings, { network_object_id: network.id, target: network.properties.nodes[0] });
  }
  return settings;
}

function capabilitySettings() {
  let value;
  try {
    value = JSON.parse(elements.capabilitySettings.value || "{}");
  } catch (error) {
    throw new Error(`Capability settings are not valid JSON: ${error.message}`);
  }
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error("Capability settings must be a JSON object.");
  }
  return value;
}

function updateCapabilitySelection({ resetSettings = true } = {}) {
  const capability = selectedCapability();
  if (!capability) {
    elements.capabilityDescription.textContent = "No installed capability is selected.";
    byId("capability-run-button").disabled = true;
    byId("add-capability-run-button").disabled = true;
    return;
  }
  elements.capabilityDescription.className = `notice ${capability.applicable ? "safe" : "neutral"}`;
  elements.capabilityDescription.innerHTML = `<strong>${escapeHtml(capability.pack_id)} · v${escapeHtml(capability.version)}</strong><br>${escapeHtml(capability.description)}<br>Backend: ${escapeHtml(capability.backend)} · outputs: ${capability.outputs.map((item) => escapeHtml(item.identifier)).join(", ")}${capability.applicable ? "" : `<br>Unavailable: ${escapeHtml(capability.inapplicable_reason)}`}`;
  if (resetSettings && state.model) {
    elements.capabilitySettings.value = JSON.stringify(
      defaultCapabilitySettings(capability, state.model), null, 2,
    );
  }
  byId("capability-run-button").disabled = !capability.applicable;
  byId("add-capability-run-button").disabled = !capability.applicable;
}

function configureCapabilityPacks(model) {
  elements.capabilitySelect.replaceChildren();
  const catalogue = model.capabilities.installed_packs || [];
  for (const capability of catalogue) {
    const option = new Option(
      `${capability.title} · ${capability.id}@${capability.version}${capability.applicable ? "" : " (not applicable)"}`,
      capability.id,
    );
    elements.capabilitySelect.add(option);
  }
  const firstApplicable = catalogue.find((item) => item.applicable);
  if (firstApplicable) elements.capabilitySelect.value = firstApplicable.id;
  elements.capabilitySelect.disabled = !catalogue.length;
  updateCapabilitySelection();
}

function renderCapabilityPlan() {
  const rows = state.capabilityPlan.map((item, index) => ({
    order: index + 1,
    capability: item.capability_id,
    version: item.capability_version,
    settings: JSON.stringify(item.settings),
  }));
  elements.capabilityPlan.className = rows.length ? "" : "empty-content";
  elements.capabilityPlan.innerHTML = rows.length
    ? tableHtml(["order", "capability", "version", "settings"], rows)
    : "No runs selected.";
  byId("clear-capability-plan-button").disabled = !rows.length;
}

function renderCapabilityResult(result) {
  state.capabilityResult = result;
  const run = result.run;
  const artifacts = result.artifacts.map((item) => ({
    type: `${item.artifact_type}@${item.artifact_type_version}`,
    artifact: shortHash(item.artifact_id),
    scientific_sha256: shortHash(item.artifact_sha256),
  }));
  elements.capabilityResult.className = "content-stack";
  const portableText = JSON.stringify(result.results, null, 2);
  const displayedText = portableText.length > 200000
    ? `${portableText.slice(0, 200000)}\n… result preview truncated; the complete typed artifact remains available to experiment storage and reproduction.`
    : portableText;
  elements.capabilityResult.innerHTML = `
    <div class="metric-row">
      ${metricHtml("Capability", run.capability_id)}
      ${metricHtml("Version", run.capability_version)}
      ${metricHtml("Workload", run.workload_units)}
      ${metricHtml("Status", run.status)}
    </div>
    ${structureCard("Typed artifacts", ["type", "artifact", "scientific_sha256"], artifacts, "No artifacts produced.")}
    <details><summary>Inspect portable scientific result</summary><pre class="json-document">${escapeHtml(displayedText)}</pre></details>
  `;
  if (result.figure) renderPlot(elements.capabilityPlot, result.figure, "model-laboratory-artifact-view");
  else {
    purgePlot(elements.capabilityPlot);
    elements.capabilityPlot.classList.add("hidden");
  }
  setChip(elements.capabilityStatus, "Completed", "success");
}

async function runSelectedCapability() {
  const capability = selectedCapability();
  if (!state.model || !capability?.applicable) throw new Error("Select an applicable installed capability.");
  const revision = state.inputRevision;
  const result = await withBusy(
    "Running capability",
    `${capability.title} is executing through its bounded local capability pack…`,
    () => engine("run_capability", {
      source: elements.source.value,
      capability_id: capability.id,
      capability_version: capability.version,
      settings: capabilitySettings(),
    }),
  );
  if (!shouldCommitResult(revision, state.inputRevision)) {
    setStatus("Capability result discarded because its inputs changed.");
    return null;
  }
  renderCapabilityResult(result);
  activateTab("capabilities");
  setStatus(`${capability.title} completed.`, `Artifact ${shortHash(result.artifacts[0]?.artifact_id)}`);
  return result;
}

function addSelectedCapabilityToPlan() {
  const capability = selectedCapability();
  if (!capability?.applicable) throw new Error("Select an applicable installed capability.");
  state.capabilityPlan.push({
    capability_id: capability.id,
    capability_version: capability.version,
    settings: capabilitySettings(),
  });
  state.inputRevision += 1;
  renderCapabilityPlan();
  clearPreparedState("The author review was cleared because the generic run plan changed.");
  renderCommittedBoundary();
  setStatus(`${capability.title} added to the ordered experiment plan.`);
}

function renderParameterControls(model, initialValues = null) {
  elements.parameterControls.replaceChildren();
  if (!model.structure.parameters.length) {
    elements.parameterControls.className = "empty-setting";
    elements.parameterControls.textContent = "This model has no adjustable parameters.";
    byId("reset-parameters-button").disabled = true;
    return;
  }
  elements.parameterControls.className = "";
  for (const parameter of model.structure.parameters) {
    const value = initialValues?.[parameter.name] ?? parameter.default;
    const wrapper = document.createElement("div");
    wrapper.className = "parameter-control";
    const header = document.createElement("div");
    header.className = "parameter-label";
    header.innerHTML = `<strong>${escapeHtml(parameter.label)}</strong><span>${escapeHtml(parameter.domain)}${parameter.unit ? ` ${escapeHtml(parameter.unit)}` : ""}</span>`;
    const inputs = document.createElement("div");
    inputs.className = "parameter-inputs";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = parameter.lower;
    slider.max = parameter.upper;
    slider.step = Math.max(Math.abs(parameter.upper - parameter.lower) / 400, Number.EPSILON);
    slider.value = value;
    slider.setAttribute("aria-label", `${parameter.label} slider`);
    const number = document.createElement("input");
    number.type = "number";
    number.min = parameter.lower;
    number.max = parameter.upper;
    number.step = "any";
    number.value = value;
    number.dataset.parameter = parameter.name;
    number.setAttribute("aria-label", `${parameter.label} value`);
    const changed = () => invalidateComputedResults(
      "parameter-values",
      "The author review was cleared because experiment inputs changed.",
    );
    slider.addEventListener("input", () => { number.value = slider.value; changed(); });
    number.addEventListener("input", () => {
      const candidate = Number(number.value);
      if (Number.isFinite(candidate)) slider.value = String(candidate);
      changed();
    });
    inputs.append(slider, number);
    wrapper.append(header, inputs);
    elements.parameterControls.append(wrapper);
  }
  byId("reset-parameters-button").disabled = false;
}

function structureCard(title, columns, rows, emptyMessage) {
  return `<section class="content-card"><h3>${escapeHtml(title)}</h3>${tableHtml(columns, rows, emptyMessage)}</section>`;
}

function renderStructure(model) {
  const structure = model.structure;
  const metrics = Object.entries(model.counts)
    .map(([name, value]) => metricHtml(name.replaceAll("_", " "), value))
    .join("");
  const variables = structure.variables.map((item) => ({
    name: item.name, label: item.label || "—", domain: item.domain,
    initial: item.initial_display, unit: item.unit || "—", description: item.description || "—",
  }));
  const parameters = structure.parameters.map((item) => ({
    name: item.name, label: item.label, default: item.default_display,
    domain: item.domain, unit: item.unit || "—", description: item.description || "—",
  }));
  const definitions = [
    ...structure.constants.map((item) => ({ kind: "constant", name: item.name, definition: item.value_display, dependencies: "—" })),
    ...structure.derived_quantities.map((item) => ({ kind: "derived", name: item.name, definition: item.definition, dependencies: item.dependencies.join(", ") || "—" })),
    ...structure.functions.map((item) => ({ kind: "function", name: item.name, definition: item.definition, dependencies: item.dependencies.join(", ") || "—" })),
    ...structure.vector_functions.map((item) => ({
      kind: `vector R^${model.counts.variables} → R^${item.output_dimension}`,
      name: item.name,
      definition: item.components.map((component) => `${component.name}=${component.definition}`).join("; "),
      dependencies: [...new Set(item.components.flatMap((component) => component.dependencies))].join(", ") || "—",
    })),
    ...structure.matrix_functions.map((item) => ({
      kind: `matrix ${item.shape.join("×")}`,
      name: item.name,
      definition: JSON.stringify(item.entries),
      dependencies: "see Model Graph",
    })),
  ];
  const assumptions = structure.assumptions.map((item) => ({ name: item.name, statement: item.statement, affects: item.affects.join(", ") || "—" }));
  const ambiguities = structure.ambiguities.map((item) => ({
    name: item.name, statement: item.statement, status: item.status,
    blocking: item.blocking ? "yes" : "no", resolution: item.resolution || "—",
  }));
  elements.structureContent.className = "content-stack";
  elements.structureContent.innerHTML = `
    <div class="metric-row">${metrics}</div>
    ${structureCard("Variables", ["name", "label", "domain", "initial", "unit", "description"], variables, "No variables.")}
    ${structureCard("Parameters", ["name", "label", "default", "domain", "unit", "description"], parameters, "No parameters.")}
    ${structureCard("Definitions", ["kind", "name", "definition", "dependencies"], definitions, "No definitions.")}
    ${structureCard("Constraints", ["name", "relation", "dependencies"], structure.constraints.map((item) => ({ ...item, dependencies: item.dependencies.join(", ") || "—" })), "No constraints.")}
    ${structureCard("Assumptions", ["name", "statement", "affects"], assumptions, "No assumptions were supplied.")}
    ${structureCard("Ambiguities", ["name", "statement", "status", "blocking", "resolution"], ambiguities, "No unresolved ambiguities.")}
    ${structureCard("Model Graph objects", ["id", "kind", "kind_version", "value_type", "executable", "opaque"], structure.model_graph.objects.map((item) => ({ ...item, value_type: `${item.value_type.kind}${item.value_type.shape.length ? `[${item.value_type.shape.join(",")}]` : ""}` })), "No graph objects.")}
    ${structureCard("Model Graph relationships", ["id", "kind", "source", "target", "opaque"], structure.model_graph.relationships, "No graph relationships.")}
    ${structureCard("Structured assets", ["id", "kind", "kind_version", "media_type", "size", "opaque"], structure.model_graph.assets, "No structured assets.")}
  `;
}

function configureVisualisations(model, selected = null) {
  elements.visualisationSelect.replaceChildren();
  for (const visualisation of model.capabilities.visualisations) {
    const option = new Option(visualisation, visualisation);
    option.selected = visualisation === (selected || model.defaults.selected_visualisation);
    elements.visualisationSelect.add(option);
  }
  elements.visualisationSelect.disabled = !model.capabilities.visualisations.length;
}

function configureSweep(model, saved = null) {
  const capable = model.capabilities.analyses.includes("parameter sweep") && model.structure.parameters.length > 0;
  elements.sweepUnavailable.classList.toggle("hidden", capable);
  byId("run-sweep-button").disabled = !capable;
  byId("include-sweep").disabled = !capable;
  elements.sweepParameter.replaceChildren();
  for (const parameter of model.structure.parameters) {
    elements.sweepParameter.add(new Option(parameter.label || parameter.name, parameter.name));
  }
  elements.sweepVisualisation.replaceChildren();
  ["stationary-point counts", "stationary-point positions", "stationary function values"].forEach((value) => elements.sweepVisualisation.add(new Option(value, value)));
  if (!capable) return;
  if (saved?.parameter_name) elements.sweepParameter.value = saved.parameter_name;
  const selectedParameter = model.structure.parameters.find((item) => item.name === elements.sweepParameter.value) || model.structure.parameters[0];
  elements.sweepStart.value = saved?.start ?? selectedParameter.lower;
  elements.sweepEnd.value = saved?.end ?? selectedParameter.upper;
  elements.sweepSteps.value = saved?.step_count ?? 21;
  if (saved?.selected_visualisation) elements.sweepVisualisation.value = saved.selected_visualisation;
}

function applyModel(model, { initialValues = null, preserveSettings = false, selectedVisualisation = null, sweep = null } = {}) {
  state.model = model;
  state.source = elements.source.value;
  renderModelIdentity(model);
  renderCapabilities(model);
  renderStructure(model);
  renderParameterControls(model, initialValues);
  configureCapabilityPacks(model);
  configureVisualisations(model, selectedVisualisation);
  configureSweep(model, sweep);
  if (!preserveSettings) applyDefaults(model);
  elements.source.classList.remove("invalid");
  setChip(elements.validationChip, "Validated", "success");
  const hasLegacyAnalysis = model.capabilities.visualisations.length > 0;
  byId("analyse-button").disabled = !(hasLegacyAnalysis || model.capabilities.installed_packs.some((item) => item.applicable));
  byId("analyse-button").textContent = hasLegacyAnalysis ? "Run analysis" : "Run capability";
  byId("prepare-experiment-button").disabled = false;
  const blocking = model.blocking_ambiguities?.length || 0;
  setStatus(blocking ? `Validated with ${blocking} blocking ambiguity resolution(s).` : "Model validated. Ready for analysis.", `Source ${shortHash(model.source_sha256)}`);
}

function numericalSettings() {
  const settings = {
    points_1d: Number.parseInt(byId("points-1d").value, 10),
    points_per_axis_2d: Number.parseInt(byId("points-2d").value, 10),
    samples_1d: Number.parseInt(byId("samples-1d").value, 10),
    seeds_per_axis_2d: Number.parseInt(byId("seeds-2d").value, 10),
    root_tolerance: Number(byId("root-tolerance").value),
    relative_tolerance: Number(byId("relative-tolerance").value),
    absolute_tolerance: Number(byId("absolute-tolerance").value),
  };
  const errors = validateNumericalSettings(settings);
  if (errors.length) throw new Error(errors.join(" "));
  return settings;
}

function parameterValues() {
  const values = {};
  document.querySelectorAll("[data-parameter]").forEach((input) => {
    const value = Number(input.value);
    if (!Number.isFinite(value)) throw new Error(`Parameter ‘${input.dataset.parameter}’ must be finite.`);
    values[input.dataset.parameter] = value;
  });
  return values;
}

function currentSweep(enabled = true) {
  if (!state.model?.capabilities.analyses.includes("parameter sweep")) return null;
  return {
    enabled,
    parameter_name: elements.sweepParameter.value,
    start: Number(elements.sweepStart.value),
    end: Number(elements.sweepEnd.value),
    step_count: Number.parseInt(elements.sweepSteps.value, 10),
    selected_visualisation: elements.sweepVisualisation.value,
  };
}

function analysisPayload({ includeConfiguredSweep = false } = {}) {
  if (!state.model) throw new Error("Validate the model before running analysis.");
  const settings = numericalSettings();
  const payload = {
    source: elements.source.value,
    interpreter_acceptances: state.interpreterAcceptances,
    parameter_values: parameterValues(),
    selected_visualisation: elements.visualisationSelect.value,
    evaluation_settings: { points_1d: settings.points_1d, points_per_axis_2d: settings.points_per_axis_2d },
    stationary_settings: { samples_1d: settings.samples_1d, seeds_per_axis_2d: settings.seeds_per_axis_2d, root_tolerance: settings.root_tolerance },
    reproduction_settings: { relative_tolerance: settings.relative_tolerance, absolute_tolerance: settings.absolute_tolerance },
  };
  if (includeConfiguredSweep && byId("include-sweep").checked) payload.sweep = currentSweep(true);
  return payload;
}

function renderDiagnostics(target, diagnostics = []) {
  target.innerHTML = diagnostics.map((item) => `
    <div class="diagnostic ${escapeHtml(item.severity)}">
      <strong>${escapeHtml(item.code)}</strong>${escapeHtml(item.message)}
      ${item.details_text && item.details_text !== "—" ? `<div class="small-note">${escapeHtml(item.details_text)}</div>` : ""}
    </div>
  `).join("");
}

function plotLayout(figure) {
  return {
    ...figure.layout,
    autosize: true,
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdfe",
    font: { ...(figure.layout?.font || {}), family: "Segoe UI, system-ui, sans-serif", color: "#244f69", size: 11 },
    margin: { l: 58, r: 32, t: 48, b: 52, ...(figure.layout?.margin || {}) },
  };
}

function renderPlot(target, figure, filename = "model-laboratory-figure") {
  if (!window.Plotly) throw new Error("The bundled Plotly renderer did not load.");
  target.classList.remove("hidden");
  window.Plotly.react(target, figure.data, plotLayout(figure), {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    toImageButtonOptions: { format: "svg", filename },
  });
}

function resizePlot(target) {
  if (window.Plotly && target?.data) window.Plotly.Plots.resize(target);
}

function renderSymbolic(symbolic) {
  if (!symbolic?.entries?.length) {
    elements.symbolicContent.className = "formula-stack empty-content";
    elements.symbolicContent.textContent = "No symbolic analysis is available for this model.";
    return;
  }
  elements.symbolicContent.className = "formula-stack";
  elements.symbolicContent.innerHTML = symbolic.entries.map((entry) => `
    <article class="formula-card">
      <h3>${escapeHtml(entry.label)}</h3>
      ${entry.mathml}
      <details><summary>LaTeX source</summary><div class="latex-source">${escapeHtml(entry.latex)}</div></details>
    </article>
  `).join("");
}

function renderStationary(stationary) {
  if (!stationary?.available) {
    setChip(elements.stationaryStatus, stationary?.error ? "Unavailable" : "Not applicable", stationary?.error ? "error" : "neutral");
    elements.stationaryContent.className = "content-stack empty-content";
    elements.stationaryContent.textContent = stationary?.error || "Stationary-point analysis is not available for this model.";
    return;
  }
  const tone = stationary.status === "complete" ? "success" : stationary.status === "partial" ? "warning" : "error";
  setChip(elements.stationaryStatus, stationary.status, tone);
  const statistics = stationary.statistics?.length
    ? `<div class="metric-row">${stationary.statistics.map((item) => metricHtml(item.measure, item.display)).join("")}</div>`
    : "";
  const diagnostics = stationary.diagnostics?.map((item) => `<div class="diagnostic ${escapeHtml(item.severity)}"><strong>${escapeHtml(item.code)}</strong>${escapeHtml(item.message)}</div>`).join("") || "";
  elements.stationaryContent.className = "content-stack";
  elements.stationaryContent.innerHTML = `${statistics}${tableHtml(stationary.columns, stationary.rows, "No stationary points were found in the declared domain.")}${diagnostics}`;
}

function renderProvenance(rows) {
  const interpreterRows = state.interpreterAcceptances.map((acceptance) => ({
    reference: `interpreter:model_source:${shortHash(acceptance.proposal_sha256)}`,
    origin: "suggested",
    source: `${acceptance.provider.provider} · ${acceptance.provider.model_tag || acceptance.provider.model}`,
    operation: "explicitly accepted local interpreter proposal",
    approval: "approved",
    note: acceptance.compiler
      ? `observed digest ${shortHash(acceptance.provider.observed_model_digest || acceptance.provider.model_digest)} · edit program ${shortHash(acceptance.compiler.edit_program_sha256)} · acceptance ${shortHash(acceptance.acceptance_sha256)}`
      : `observed digest ${shortHash(acceptance.provider.observed_model_digest || acceptance.provider.model_digest)} · legacy whole-source acceptance ${shortHash(acceptance.acceptance_sha256)}`,
  }));
  elements.provenanceContent.className = "content-stack";
  elements.provenanceContent.innerHTML = tableHtml(["reference", "origin", "source", "operation", "approval", "note"], [...interpreterRows, ...rows], "No provenance records.");
}

function renderWorkload(workload) {
  if (!workload) return;
  elements.workloadContent.className = "";
  const heading = workload.within_budget
    ? '<div class="workload-good">Within configured safety budget</div>'
    : '<div class="workload-bad">Exceeds configured safety budget</div>';
  const rows = workload.rows.map((row) => `<li><span>${escapeHtml(row.component)}</span><strong>${escapeHtml(row.value)}</strong></li>`).join("");
  const violations = workload.violations.map((value) => `<div class="diagnostic error">${escapeHtml(value)}</div>`).join("");
  elements.workloadContent.innerHTML = `${heading}<ul class="workload-list">${rows}</ul>${violations}`;
}

function renderAnalysis(result) {
  state.analysis = result;
  state.model = result.model;
  elements.plotEmpty.classList.add("hidden");
  renderPlot(elements.modelPlot, result.figure, defaultExperimentName(result.model.name, "-model"));
  renderDiagnostics(elements.evaluationDiagnostics, result.evaluation.diagnostics);
  renderSymbolic(result.symbolic);
  renderStationary(result.stationary);
  renderProvenance(result.provenance);
  renderWorkload(result.workload);
  setStatus(`Analysis complete: ${result.evaluation.status}.`, result.selected_visualisation);
}

function renderSweep(result) {
  state.sweep = result;
  elements.sweepResults.classList.remove("hidden");
  elements.sweepMetrics.innerHTML = [
    metricHtml("Steps", result.step_count),
    metricHtml("Successful", result.successful_step_count),
    metricHtml("Partial", result.partial_step_count),
    metricHtml("Failed", result.failed_step_count),
  ].join("");
  renderPlot(elements.sweepPlot, result.figure, defaultExperimentName(state.model?.name, "-sweep"));
  const diagnosticTable = result.diagnostics?.length
    ? `<section class="content-card"><h3>Step diagnostics</h3>${tableHtml([result.parameter_name, "status", "code", "severity", "message", "details_text"], result.diagnostics)}</section>`
    : "";
  elements.sweepSummary.innerHTML = `<section class="content-card"><h3>Sweep summary</h3>${tableHtml(result.summary_columns, result.summary_rows)}</section>${diagnosticTable}`;
  renderWorkload(result.workload);
  setStatus(`Parameter sweep complete: ${result.successful_step_count}/${result.step_count} steps successful.`);
}

async function validateCurrentModel({ initialValues = null, preserveSettings = false, selectedVisualisation = null, sweep = null } = {}) {
  const source = elements.source.value;
  const revision = state.inputRevision;
  const model = await engine("inspect_model", { source });
  if (!shouldCommitResult(revision, state.inputRevision)) {
    setStatus("Model validation result discarded because the source changed.");
    return null;
  }
  clearAnalysisViews();
  clearPreparedState();
  applyModel(model, { initialValues, preserveSettings, selectedVisualisation, sweep });
  return model;
}

async function runAnalysis() {
  const revision = state.inputRevision;
  const payload = analysisPayload();
  const result = await withBusy("Running analysis", "Evaluating the model and deriving analytical structure…", () => engine("analyse_model", payload));
  if (!shouldCommitResult(revision, state.inputRevision)) {
    setStatus("Analysis result discarded because its inputs changed.");
    return null;
  }
  renderAnalysis(result);
  return result;
}

async function runPrimaryAnalysis() {
  if (state.model?.capabilities.visualisations.length) return runAnalysis();
  return runSelectedCapability();
}

async function runSweep() {
  const revision = state.inputRevision;
  const payload = analysisPayload();
  payload.sweep = currentSweep(true);
  const result = await withBusy("Running parameter sweep", "Each parameter coordinate is analysed under the recorded numerical settings…", () => engine("run_sweep", payload));
  if (!shouldCommitResult(revision, state.inputRevision)) {
    setStatus("Parameter-sweep result discarded because its inputs changed.");
    return null;
  }
  renderSweep(result);
  return result;
}

function workloadHtml(workload) {
  const label = workload.within_budget ? "Within safety budget" : "Exceeds safety budget";
  return `<div class="metric-row">${metricHtml("Workload", label)}${workload.rows.slice(0, 3).map((item) => metricHtml(item.component, item.value)).join("")}</div>`;
}

function renderExperimentInspection(inspection) {
  const model = inspection.model;
  const environment = tableHtml(["component", "value"], inspection.environment, "No environment record.");
  const fingerprints = tableHtml(["name", "sha256"], inspection.recorded_results, "No result fingerprints recorded.");
  const approval = inspection.author_approved_for_publication ? "Author-approved" : "Draft / not frozen";
  const runRows = inspection.run_protocol?.runs?.map((item) => ({
    capability: `${item.capability_id}@${item.capability_version}`,
    workload: item.workload_units,
    status: item.status,
    artifacts: item.artifact_ids.length,
    backend: item.backend_identity?.backend || "legacy",
  })) || [];
  elements.openedExperiment.classList.remove("hidden");
  elements.openedExperiment.innerHTML = `
    <section class="content-card">
      <div class="split-heading">
        <div><span class="eyebrow">Validated without execution</span><h3>${escapeHtml(model.name)}</h3></div>
        <span class="status-chip ${inspection.workload.within_budget ? "success" : "error"}">${escapeHtml(approval)}</span>
      </div>
      <div class="metric-row">
        ${metricHtml("Format", inspection.bundle_format ? `${inspection.bundle_format.name} ${inspection.bundle_format.version}` : inspection.kind)}
        ${metricHtml("Laboratory", inspection.saved_laboratory_version)}
        ${metricHtml("Experiment ID", shortHash(inspection.experiment_id))}
        ${metricHtml("State SHA-256", shortHash(inspection.state_sha256))}
        ${metricHtml("AI acceptances", (inspection.interpreter_acceptances || []).length)}
      </div>
      ${workloadHtml(inspection.workload)}
      <div class="experiment-actions"><button class="button primary" id="reproduce-opened-button" type="button" ${inspection.workload.within_budget ? "" : "disabled"}>Reproduce experiment</button></div>
      <details><summary>Recorded result fingerprints</summary>${fingerprints}</details>
      ${runRows.length ? `<details><summary>Typed run records</summary>${tableHtml(["capability", "workload", "status", "artifacts", "backend"], runRows)}</details>` : ""}
      <details><summary>Recorded environment</summary>${environment}</details>
    </section>
  `;
  byId("reproduce-opened-button")?.addEventListener("click", () => reproduceOpenedExperiment().catch(() => {}));
  setChip(elements.experimentStatus, "Inspected safely", "success");
}

async function inspectExperiment(document) {
  const inspection = await withBusy("Validating experiment", "Checking bundle integrity and reconstructing state without executing analyses…", () => engine("inspect_experiment", {
    filename: document.name,
    data_base64: document.dataBase64,
  }));
  state.openedDocument = document;
  state.experimentInspection = inspection;
  state.reproduction = null;
  setSource(inspection.model_source, document.name, document.path);
  state.interpreterAcceptances = inspection.interpreter_acceptances || [];
  applyModel(inspection.model, {
    initialValues: inspection.parameter_values,
    preserveSettings: true,
    selectedVisualisation: inspection.selected_visualisation,
    sweep: inspection.sweep,
  });
  renderProvenance(inspection.model.provenance || []);
  const evaluation = inspection.evaluation_settings;
  const stationary = inspection.stationary_settings;
  const reproduction = inspection.reproduction_settings;
  byId("points-1d").value = evaluation.points_1d;
  byId("points-2d").value = evaluation.points_per_axis_2d;
  byId("samples-1d").value = stationary.samples_1d;
  byId("seeds-2d").value = stationary.seeds_per_axis_2d;
  byId("root-tolerance").value = stationary.root_tolerance;
  byId("relative-tolerance").value = reproduction.relative_tolerance;
  byId("absolute-tolerance").value = reproduction.absolute_tolerance;
  renderWorkload(inspection.workload);
  renderExperimentInspection(inspection);
  activateTab("experiment");
  setStatus("Experiment validated and inspected. No computation was performed.", `State ${shortHash(inspection.state_sha256)}`);
}

function diagnosticsRows(records, origin) {
  return (records || []).map((item) => ({
    origin,
    code: item.code,
    severity: item.severity,
    details: Array.isArray(item.details) ? item.details.map((entry) => `${entry.key ?? entry[0]}=${entry.value ?? entry[1]}`).sort().join(", ") : "—",
    message: item.message || "—",
  }));
}

function renderReproduction(result) {
  state.reproduction = result;
  const report = result.report;
  const tone = reportTone(report.status);
  const interpreterAcceptanceCount = report.provenance?.interpreter_acceptances?.length || 0;
  const resultRows = report.results.map((item) => ({
    result: item.name,
    status: item.status,
    "strict fingerprint": item.strict.matches ? "identical" : "different",
    "numerical comparison": item.numerical.reference_available ? (item.numerical.matches ? "within tolerance" : "outside tolerance") : "not available",
    "maximum absolute deviation": scientificDisplay(item.numerical.maximum_absolute_deviation),
    "maximum relative deviation": scientificDisplay(item.numerical.maximum_relative_deviation),
  }));
  const diagnosticRows = report.results.flatMap((item) => [
    ...diagnosticsRows(item.diagnostics?.saved, `${item.name} · saved`),
    ...diagnosticsRows(item.diagnostics?.current, `${item.name} · current`),
  ]);
  elements.reproductionContent.className = "content-stack";
  elements.reproductionContent.innerHTML = `
    <section class="report-hero ${tone}">
      <span class="eyebrow">Final reproduction status</span>
      <h3>${escapeHtml(report.status)}</h3>
      <p>Experiment ${escapeHtml(report.experiment_id || "legacy state")} · state ${escapeHtml(shortHash(report.experiment_state_sha256))}</p>
    </section>
    <div class="report-grid">
      <section class="content-card"><h3>Mathematical identity</h3><div class="metric-row">
        ${metricHtml("Model validated", report.model.validated ? "Yes" : "No")}
        ${metricHtml("Source integrity", report.model.source_integrity ? "Identical" : "Different")}
        ${metricHtml("Canonical Model IR", report.model.canonical_ir_matches ? "Identical" : "Different")}
        ${metricHtml("Exact results", `${report.summary.exact_result_count}/${report.summary.result_count}`)}
        ${metricHtml("Statistical results", report.summary.statistical_result_count || 0)}
        ${metricHtml("AI acceptances", interpreterAcceptanceCount)}
      </div></section>
      <section class="content-card"><h3>Environment and tolerance</h3><div class="metric-row">
        ${metricHtml("Environment", report.environment.status)}
        ${metricHtml("Laboratory version", report.laboratory.current_version)}
        ${metricHtml("rtol", scientificDisplay(report.numerical_tolerances.relative_tolerance))}
        ${metricHtml("atol", scientificDisplay(report.numerical_tolerances.absolute_tolerance))}
      </div></section>
    </div>
    <section class="content-card"><h3>Analysis-by-analysis results</h3>${tableHtml(["result", "status", "strict fingerprint", "numerical comparison", "maximum absolute deviation", "maximum relative deviation"], resultRows)}</section>
    ${report.environment.differences.length ? `<section class="content-card"><h3>Environment differences</h3>${report.environment.differences.map((item) => `<div class="diagnostic warning">${escapeHtml(item)}</div>`).join("")}</section>` : ""}
    ${diagnosticRows.length ? `<section class="content-card"><h3>Diagnostics</h3><p class="small-note">Code, severity, and canonical details define scientific diagnostic identity. Message is preserved as human presentation.</p>${tableHtml(["origin", "code", "severity", "details", "message"], diagnosticRows)}</section>` : ""}
    ${result.figure ? '<section class="content-card"><h3>Reproduced visualisation</h3><div class="plot-surface" id="reproduction-plot"></div></section>' : ""}
    <div class="report-actions"><button class="button secondary" id="export-report-json" type="button">Export JSON report</button><button class="button secondary" id="export-report-text" type="button">Export text report</button></div>
  `;
  if (result.figure) renderPlot(byId("reproduction-plot"), result.figure, "model-laboratory-reproduction");
  byId("export-report-json").addEventListener("click", () => saveEncoded("report-json", "reproduction-report.json", textToBase64(result.report_json)).catch(() => {}));
  byId("export-report-text").addEventListener("click", () => saveEncoded("report-text", "reproduction-report.txt", textToBase64(result.report_text)).catch(() => {}));
  activateTab("reproduction");
  setStatus(report.status, `Environment ${report.environment.status}`);
}

async function reproduceOpenedExperiment() {
  if (!state.openedDocument) throw new Error("Open and inspect an experiment first.");
  const revision = state.inputRevision;
  const openedDocument = state.openedDocument;
  const result = await withBusy("Reproducing experiment", "Executing the frozen analyses under the recorded workload and tolerance policy…", () => engine("reproduce_experiment", {
    filename: openedDocument.name,
    data_base64: openedDocument.dataBase64,
  }));
  if (!shouldCommitResult(revision, state.inputRevision) || state.openedDocument !== openedDocument) {
    setStatus("Reproduction result discarded because the opened experiment changed.");
    return;
  }
  renderReproduction(result);
}

function reviewMetrics(review) {
  const experiment = review.experiment;
  const runLabels = review.run_records?.map((item) => item.capability_id)
    || experiment.analyses?.map((item) => item.name)
    || [];
  const tolerances = experiment.reproduction_policy
    || experiment.numerical_settings?.reproduction_tolerances
    || {};
  return `
    <div class="metric-row">
      ${metricHtml("Experiment ID", shortHash(experiment.experiment_id))}
      ${metricHtml("State", shortHash(experiment.experiment_state_sha256))}
      ${metricHtml("Model IR", shortHash(experiment.model.canonical_ir_sha256))}
      ${metricHtml("Runs", runLabels.join(", "))}
      ${metricHtml("rtol", scientificDisplay(tolerances.relative_tolerance))}
      ${metricHtml("atol", scientificDisplay(tolerances.absolute_tolerance))}
    </div>
  `;
}

function renderPrepared(result) {
  state.prepared = result;
  elements.authorReview.classList.remove("hidden");
  elements.reviewHash.textContent = result.review_sha256;
  elements.reviewHash.title = result.review_sha256;
  elements.reviewSummary.innerHTML = reviewMetrics(result.review);
  elements.reviewJson.textContent = JSON.stringify(result.review, null, 2);
  elements.approveReview.checked = false;
  byId("save-draft-button").disabled = false;
  byId("save-state-button").disabled = false;
  byId("publish-experiment-button").disabled = true;
  setChip(elements.experimentStatus, "Review required", "warning");
  renderCommittedBoundary();
  elements.authorReview.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function prepareExperiment() {
  return prepareRunExperiment();
}

function defaultExperimentPlan() {
  if (!state.model) return [];
  const preferred = [
    state.model.counts.variables === 1
      ? "org.modellab.scalar.evaluate-curve"
      : "org.modellab.scalar.evaluate-surface",
    "org.modellab.scalar.differential",
    "org.modellab.scalar.stationary-points",
    "org.modellab.vector.differential",
    "org.modellab.vector.solve-roots",
    "org.modellab.vector.field-2d",
    "org.modellab.matrix.analyse",
    "org.modellab.optimization.constrained",
  ];
  return preferred.flatMap((identifier) => {
    const capability = state.model.capabilities.installed_packs.find(
      (item) => item.id === identifier && item.applicable,
    );
    return capability ? [{
      capability_id: capability.id,
      capability_version: capability.version,
      settings: defaultCapabilitySettings(capability, state.model),
    }] : [];
  });
}

async function prepareRunExperiment() {
  const revision = state.inputRevision;
  if (!state.model) throw new Error("Validate the model before preparing an experiment.");
  const settings = numericalSettings();
  const parameters = parameterValues();
  const plan = (state.capabilityPlan.length ? state.capabilityPlan : defaultExperimentPlan())
    .map((item) => {
      const capability = state.model.capabilities.installed_packs.find(
        (candidate) => candidate.id === item.capability_id,
      );
      const runSettings = structuredClone(item.settings);
      if (capability?.settings_schema?.properties?.parameter_values) {
        runSettings.parameter_values = parameters;
      }
      return { ...item, settings: runSettings };
    });
  if (!plan.length) throw new Error("Select at least one applicable capability for the reference state.");
  const payload = {
    source: elements.source.value,
    parameter_values: parameters,
    runs: plan,
    interpreter_acceptances: state.interpreterAcceptances,
    reproduction_settings: {
      relative_tolerance: settings.relative_tolerance,
      absolute_tolerance: settings.absolute_tolerance,
    },
  };
  const result = await withBusy("Preparing reference state", "Executing the ordered capability plan and freezing typed scientific artifacts…", () => engine("prepare_run_experiment", payload));
  if (!shouldCommitResult(revision, state.inputRevision)) {
    setStatus("Prepared reference state discarded because its inputs changed.");
    return;
  }
  renderPrepared(result);
  activateTab("experiment");
  setStatus("Reference state prepared. Review and explicitly approve it before publication.", `Review ${shortHash(result.review_sha256)}`);
}

async function commitPreparedExperiment() {
  if (!state.prepared) throw new Error("Prepare an experiment state before committing it.");
  const revision = state.inputRevision;
  const result = await withBusy(
    "Committing experiment state",
    "Validating the immutable state envelope and separating it from the live authoring state…",
    () => engine("commit_experiment_state", {
      state_json: state.prepared.state_json,
      expected_state_sha256: state.prepared.state_sha256,
      parent_commit_sha256: state.committed?.commit_sha256 || null,
    }),
  );
  if (!shouldCommitResult(revision, state.inputRevision)) {
    setStatus("Commit discarded because the live experiment changed before the commit completed.");
    return;
  }
  state.committed = result;
  state.committedInputRevision = revision;
  renderCommittedBoundary();
  setStatus(
    "Prepared experiment state committed. Live editing may continue without mutating this snapshot.",
    `Commit ${shortHash(result.commit_sha256)}`,
  );
  toast("Experiment state committed.");
}

async function saveCommittedReceipt() {
  if (!state.committed?.commit_json) throw new Error("There is no committed state to save.");
  const name = defaultExperimentName(state.model?.name || "experiment", "-commit.json");
  await saveEncoded("report-json", name, textToBase64(state.committed.commit_json));
}

async function openCommittedReceipt() {
  const document = await native("open_document", { kind: "commit" });
  if (!document) return;
  const commitJson = base64ToText(document.dataBase64);
  const inspected = await withBusy(
    "Validating commit receipt",
    "Checking the immutable receipt and reconstructing its live authoring state without executing analyses…",
    () => engine("inspect_committed_experiment", { commit_json: commitJson }),
  );
  state.openedDocument = document;
  state.experimentInspection = null;
  state.reproduction = null;
  elements.openedExperiment.classList.add("hidden");
  setSource(inspected.model_source, document.name, document.path);
  state.interpreterAcceptances = inspected.interpreter_acceptances || [];
  applyModel(inspected.model, {
    initialValues: inspected.parameter_values,
    preserveSettings: true,
    selectedVisualisation: inspected.selected_model_visualisation,
    sweep: inspected.sweep,
  });
  renderProvenance(inspected.model.provenance || []);
  byId("points-1d").value = inspected.evaluation_settings.points_1d;
  byId("points-2d").value = inspected.evaluation_settings.points_per_axis_2d;
  byId("samples-1d").value = inspected.stationary_settings.samples_1d;
  byId("seeds-2d").value = inspected.stationary_settings.seeds_per_axis_2d;
  byId("root-tolerance").value = inspected.stationary_settings.root_tolerance;
  byId("relative-tolerance").value = inspected.reproduction_settings.relative_tolerance;
  byId("absolute-tolerance").value = inspected.reproduction_settings.absolute_tolerance;
  state.committed = inspected;
  state.committedInputRevision = state.inputRevision;
  renderCommittedBoundary();
  activateTab("experiment");
  setStatus(
    "Committed experiment receipt validated and reopened. No computation was performed.",
    `Commit ${shortHash(inspected.commit_sha256)}`,
  );
}

async function finalizeExperiment() {
  if (!state.prepared || !elements.approveReview.checked) throw new Error("Approve the exact canonical review before publication.");
  const result = await withBusy("Freezing experiment", "Rechecking exact reproduction before creating the author-approved bundle…", () => engine("finalize_experiment", {
    state_json: state.prepared.state_json,
    approved_review_sha256: state.prepared.review_sha256,
  }));
  const saved = await saveEncoded("experiment", defaultExperimentName(state.model?.name), result.bundle_base64);
  if (saved) {
    setChip(elements.experimentStatus, "Frozen and published", "success");
    setStatus("Publishable experiment saved after exact reference-state verification.", `Experiment ${shortHash(result.experiment_id)}`);
    toast("Author-approved .mlab saved.");
  }
}

async function saveEncoded(kind, suggestedName, dataBase64) {
  const path = await native("save_document", { kind, suggestedName, dataBase64 });
  if (path) setStatus(`Saved ${path}`);
  return path;
}

async function openModel() {
  const document = await native("open_document", { kind: "model" });
  if (!document) return;
  const source = base64ToText(document.dataBase64);
  state.openedDocument = null;
  state.experimentInspection = null;
  elements.openedExperiment.classList.add("hidden");
  setSource(source, document.name, document.path);
  await withBusy("Validating model", "Parsing the supplied model and constructing canonical Model IR…", () => validateCurrentModel());
}

async function openExperiment() {
  const document = await native("open_document", { kind: "experiment" });
  if (!document) return;
  await inspectExperiment({ name: document.name, path: document.path, dataBase64: document.dataBase64 });
}

async function saveModel() {
  const name = state.sourceName.match(/\.ya?ml$/i) ? state.sourceName : defaultExperimentName(state.model?.name || "model", ".yaml");
  const path = await saveEncoded("model", name, textToBase64(elements.source.value));
  if (path) toast("Model source saved.");
}

function formatByteSize(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "unknown size";
  return `${(bytes / (1024 ** 3)).toFixed(2)} GB`;
}

function syncInterpreterControls() {
  const ready = state.interpreterRuntime?.status === "ready";
  const missing = state.interpreterRuntime?.status === "model_missing"
    && state.interpreterRuntime?.model_role !== "approved_candidate";
  byId("interpreter-refresh-button").disabled = state.interpreterBusy;
  elements.interpreterAttachButton.disabled = state.interpreterBusy
    || state.interpreterAttachments.length >= 8;
  elements.interpreterAttachmentList.querySelectorAll("button").forEach((button) => {
    button.disabled = state.interpreterBusy;
  });
  byId("interpreter-request-button").disabled = state.interpreterBusy
    || !ready
    || !elements.interpreterInstruction.value.trim()
    || state.pendingInterpreterQuestion !== null;
  byId("interpreter-clarification-continue").disabled = state.interpreterBusy
    || !ready
    || !state.pendingInterpreterQuestion
    || !elements.interpreterClarificationAnswer.value.trim();
  byId("interpreter-clarification-cancel").disabled = state.interpreterBusy;
  byId("interpreter-pull-button").classList.toggle("hidden", !missing);
  byId("interpreter-pull-button").disabled = state.interpreterBusy;
  byId("interpreter-cancel-button").classList.toggle("hidden", !state.interpreterBusy);
}

function setInterpreterBusy(busy, operation = "local operation") {
  state.interpreterBusy = busy;
  if (busy) {
    elements.interpreterStatus.className = "interpreter-status";
    elements.interpreterStatus.querySelector("strong").textContent = `${operation} in progress`;
  }
  syncInterpreterControls();
}

function renderInterpreterStatus(status) {
  state.interpreterRuntime = status;
  const setIdentity = (element, digest) => {
    element.textContent = digest ? shortHash(digest, 13, 9) : "—";
    if (digest) element.title = digest;
    else element.removeAttribute("title");
  };
  elements.interpreterCandidateLifecycle.textContent = status.model_role === "approved_candidate"
    ? (status.candidate_lifecycle_state || "approved")
    : "Frozen base";
  setIdentity(elements.interpreterTrainingReport, status.training_report_sha256);
  setIdentity(elements.interpreterEvaluationReport, status.candidate_evaluation_report_sha256);
  const label = elements.interpreterStatus.querySelector("strong");
  elements.interpreterStatus.className = `interpreter-status ${status.status === "ready" ? "ready" : "unavailable"}`;
  if (status.status === "ready") {
    label.textContent = "Ready for local proposals";
    elements.interpreterRuntime.textContent = `Ollama ${status.runtime_version} · loopback only`;
    elements.interpreterRuntimeMessage.textContent = status.model_role === "approved_candidate"
      ? `The approved candidate (${formatByteSize(status.model_size_bytes)}) is active. Its local manifest and every referenced blob were SHA-256 verified; the GGUF, Stage-7 report, and one-shot promotion evidence match the registry.`
      : `The configured ${formatByteSize(status.model_size_bytes)} frozen base is installed. Its local manifest, config, GGUF, template, licence, and parameters were verified against the complete bundled lock.`;
    elements.interpreterModelDigest.textContent = shortHash(status.observed_model_digest, 13, 9);
    elements.interpreterModelDigest.title = status.observed_model_digest;
  } else if (status.status === "model_missing") {
    label.textContent = status.model_role === "approved_candidate"
      ? "Approved candidate not installed"
      : "Configured model tag not installed";
    elements.interpreterRuntime.textContent = "Ollama available · loopback only";
    elements.interpreterRuntimeMessage.textContent = status.message || (status.model_role === "approved_candidate"
      ? "Reinstall the exact exported candidate or roll back through the candidate registry."
      : "Pull the configured versioned tag before requesting a proposal.");
    elements.interpreterModelDigest.textContent = "—";
    elements.interpreterModelDigest.removeAttribute("title");
  } else {
    label.textContent = "Ollama unavailable";
    elements.interpreterRuntime.textContent = "Ollama not reachable · 127.0.0.1:11434";
    elements.interpreterRuntimeMessage.textContent = status.message || "Install and start Ollama, then refresh this panel.";
    elements.interpreterModelDigest.textContent = "—";
    elements.interpreterModelDigest.removeAttribute("title");
  }
  elements.interpreterModel.textContent = status.model_tag || "qwen3:4b-instruct-2507-q4_K_M";
  syncInterpreterControls();
}

async function refreshInterpreterStatus() {
  const status = await native("interpreter_status");
  renderInterpreterStatus(status);
  return status;
}

async function pullInterpreterModel() {
  const confirmed = window.confirm(
    "Download qwen3:4b-instruct-2507-q4_K_M through the local Ollama service? The quantized model is approximately 2.5 GB. Model Laboratory will not start inference until the download finishes.",
  );
  if (!confirmed) return;
  setInterpreterBusy(true, "model download");
  setStatus("Downloading the configured local interpreter model tag…");
  try {
    await native("interpreter_pull_model", { confirmed: true });
    toast("Configured local Qwen model tag installed.");
    setStatus("Local interpreter model installed. Ready for proposals.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message);
    toast(message, "error");
  } finally {
    setInterpreterBusy(false);
    await refreshInterpreterStatus().catch(() => {});
  }
}

function renderInterpreterProposal(result, telemetry = null) {
  state.interpreterProposal = result;
  const proposal = result.proposal;
  const provider = proposal.provider;
  const model = result.model;
  elements.interpreterProposal.classList.remove("hidden");
  elements.interpreterProposalHash.textContent = proposal.proposal_sha256;
  elements.interpreterProposalHash.title = proposal.proposal_sha256;
  elements.interpreterProposalMetrics.innerHTML = `
    <div class="metric-row">
      ${metricHtml("Validated model", model.name)}
      ${metricHtml("Source SHA-256", shortHash(model.source_sha256))}
      ${metricHtml("Canonical Model IR", shortHash(proposal.proposed_model.canonical_model_ir_sha256))}
      ${metricHtml("Compiled edits", proposal.compiler.operation_count)}
      ${metricHtml("Context round", proposal.compiler.context_round + 1)}
      ${metricHtml("Provider", `${provider.provider} ${provider.runtime_version}`)}
      ${metricHtml("Configured model tag", provider.model_tag)}
      ${metricHtml("Model role", provider.model_role.replaceAll("_", " "))}
      ${metricHtml("Observed digest", shortHash(provider.observed_model_digest))}
      ${metricHtml("Verified GGUF", shortHash(provider.artifact_evidence.model_blob_sha256))}
      ${metricHtml("Artifact lock", shortHash(provider.artifact_evidence.artifact_lock_sha256))}
      ${metricHtml("Clarification chain", `${proposal.dialogue.turn_count} turn${proposal.dialogue.turn_count === 1 ? "" : "s"}`)}
      ${metricHtml("Context profile", `${provider.generation_options.num_ctx.toLocaleString()} / ${provider.generation_options.num_predict.toLocaleString()}`)}
      ${telemetry ? metricHtml("Inference rounds", telemetry.inference_rounds) : ""}
      ${telemetry ? metricHtml("Generated tokens", telemetry.generated_tokens) : ""}
    </div>
  `;
  elements.interpreterExplanation.textContent = proposal.explanation || "No explanation was supplied.";
  elements.interpreterWarningsSection.classList.toggle("hidden", !proposal.warnings.length);
  elements.interpreterWarnings.innerHTML = proposal.warnings
    .map((warning) => `<div class="diagnostic warning">${escapeHtml(warning)}</div>`)
    .join("");
  elements.interpreterDiff.textContent = result.diff || "No source differences.";
  const assumptionRows = model.structure.assumptions.map((item) => ({
    name: item.name,
    statement: item.statement,
    affects: item.affects.join(", ") || "—",
  }));
  const ambiguityRows = model.structure.ambiguities.map((item) => ({
    name: item.name,
    statement: item.statement,
    blocking: item.blocking ? "yes" : "no",
    resolution: item.resolution || "unresolved",
  }));
  elements.interpreterModelSummary.innerHTML = `
    <div class="metric-row">
      ${metricHtml("Variables", model.counts.variables)}
      ${metricHtml("Parameters", model.counts.parameters)}
      ${metricHtml("Functions", model.counts.functions)}
      ${metricHtml("Blocking ambiguities", model.blocking_ambiguities.length)}
    </div>
    ${structureCard("Assumptions", ["name", "statement", "affects"], assumptionRows, "No assumptions were proposed.")}
    ${structureCard("Ambiguities", ["name", "statement", "blocking", "resolution"], ambiguityRows, "No ambiguities were proposed.")}
  `;
  elements.approveInterpreterProposal.checked = false;
  byId("interpreter-accept-button").disabled = true;
  elements.interpreterProposal.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function requestInterpreterProposal({ resumeClarification = false } = {}) {
  const editorInstruction = elements.interpreterInstruction.value.trim();
  const editorSource = elements.source.value;
  const activeAttachmentIds = interpreterAttachmentIds();
  if (resumeClarification) {
    if (!state.interpreterConversation
      || state.interpreterConversation.instruction !== editorInstruction
      || state.interpreterConversation.currentModelSource !== editorSource
      || !sameStringArray(state.interpreterConversation.attachmentIds || [], activeAttachmentIds)) {
      clearInterpreterClarification(true);
      throw new Error("The instruction, model source, or reference files changed during clarification. Start a new request.");
    }
  } else {
    clearInterpreterClarification(true);
    state.interpreterConversation = {
      instruction: editorInstruction,
      currentModelSource: editorSource,
      conversationId: crypto.randomUUID(),
      providerKey: null,
      lineage: [],
      attachmentIds: activeAttachmentIds,
    };
  }
  const instruction = state.interpreterConversation?.instruction || editorInstruction;
  if (!instruction) throw new Error("Describe the model or requested change first.");
  const currentModelSource = state.interpreterConversation?.currentModelSource ?? editorSource;
  const clarificationHistory = state.interpreterClarificationHistory.map((turn) => ({ ...turn }));
  clearInterpreterClarification(false);
  clearInterpreterProposal();
  setInterpreterBusy(true, "local inference");
  setStatus("Preparing a deterministic bounded model context…");
  try {
    const prepared = await engine("prepare_interpreter_context", {
      instruction,
      current_model_source: currentModelSource,
      clarification_history: clarificationHistory,
      attachment_ids: state.interpreterConversation.attachmentIds,
    });
    if (prepared.status === "unable") {
      clearInterpreterProposal();
      setStatus("The request exceeds the installed scientific capabilities.", prepared.explanation);
      toast(prepared.explanation || "The requested scientific operation is not installed.");
      return;
    }
    if (prepared.status !== "ready") throw new Error("The deterministic interpreter boundary returned an unknown status.");
    let context = prepared.context;
    let providerIdentity = null;
    let providerKey = state.interpreterConversation.providerKey;
    const telemetry = {
      inference_rounds: 0,
      prompt_tokens: 0,
      generated_tokens: 0,
      total_duration_nanoseconds: 0,
      load_duration_nanoseconds: 0,
    };

    for (let round = 0; round < prepared.maximum_context_rounds; round += 1) {
      setStatus(`Qwen is producing a typed edit program · context round ${round + 1}/${prepared.maximum_context_rounds}…`);
      const inference = await native("interpreter_request", {
        instruction,
        contextPackage: context,
      });
      if (elements.source.value !== currentModelSource
        || !sameStringArray(interpreterAttachmentIds(), state.interpreterConversation.attachmentIds)) {
        throw new Error("The model source or reference files changed during inference. The stale proposal was discarded.");
      }
      const { generation_options: _roundOptions, ...stableProviderIdentity } = inference.provider_identity;
      const identityKey = JSON.stringify(stableProviderIdentity);
      if (providerKey !== null && identityKey !== providerKey) {
        throw new Error("The local runtime, configured tag, or observed model digest changed between interpreter rounds.");
      }
      providerKey = identityKey;
      state.interpreterConversation.providerKey = identityKey;
      providerIdentity = inference.provider_identity;
      telemetry.inference_rounds += 1;
      telemetry.prompt_tokens += Number(inference.telemetry?.prompt_tokens || 0);
      telemetry.generated_tokens += Number(inference.telemetry?.generated_tokens || 0);
      telemetry.total_duration_nanoseconds += Number(inference.telemetry?.total_duration_nanoseconds || 0);
      telemetry.load_duration_nanoseconds += Number(inference.telemetry?.load_duration_nanoseconds || 0);

      const processed = await engine("process_interpreter_output", {
        instruction,
        current_model_source: currentModelSource,
        context,
        output: inference.output,
        provider_identity: providerIdentity,
        clarification_history: clarificationHistory,
        attachment_ids: state.interpreterConversation.attachmentIds,
        conversation_id: state.interpreterConversation.conversationId,
        conversation_lineage: state.interpreterConversation.lineage,
      });
      if (elements.source.value !== currentModelSource
        || !sameStringArray(interpreterAttachmentIds(), state.interpreterConversation.attachmentIds)) {
        throw new Error("The model source or reference files changed during edit compilation. The stale proposal was discarded.");
      }
      if (processed.status === "context_required") {
        context = processed.context;
        setStatus(`Qwen requested exact additional model context · preparing round ${round + 2}…`);
        continue;
      }
      if (processed.status === "needs_clarification") {
        clearInterpreterProposal();
        const question = processed.question || "The interpreter needs clarification before it can construct a proposal.";
        if (state.interpreterClarificationHistory.length >= 8) {
          throw new Error("The interpreter exhausted its bounded eight-turn clarification dialogue.");
        }
        if (!processed.turn_evidence) {
          throw new Error("The deterministic compiler did not return clarification-turn provenance.");
        }
        state.interpreterConversation.lineage.push(processed.turn_evidence);
        renderInterpreterClarification(question);
        setStatus("The interpreter needs clarification from you.", question);
        return;
      }
      if (processed.status === "unable") {
        clearInterpreterProposal();
        setStatus("The interpreter could not construct a safe edit program.", processed.explanation);
        toast(processed.explanation || "The interpreter could not construct a safe proposal.");
        return;
      }
      if (processed.status !== "proposal") {
        throw new Error("The deterministic interpreter compiler returned an unknown status.");
      }
      renderInterpreterProposal(processed, telemetry);
      clearInterpreterClarification(false);
      setStatus("Interpreter edits compiled and mathematically validated. Review the exact diff before accepting.", `Proposal ${shortHash(processed.proposal.proposal_sha256)}`);
      return;
    }
    throw new Error("The interpreter exhausted its bounded context rounds without producing a proposal.");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    clearInterpreterProposal();
    setStatus(message);
    toast(message, "error");
  } finally {
    setInterpreterBusy(false);
    await refreshInterpreterStatus().catch(() => {});
  }
}

async function continueInterpreterClarification() {
  const question = state.pendingInterpreterQuestion;
  const answer = elements.interpreterClarificationAnswer.value.trim();
  if (!question || !answer) {
    throw new Error("Answer the interpreter's clarification question first.");
  }
  state.interpreterClarificationHistory.push({ question, answer });
  state.pendingInterpreterQuestion = null;
  await requestInterpreterProposal({ resumeClarification: true });
}

async function acceptInterpreterProposal() {
  const reviewed = state.interpreterProposal;
  if (!reviewed || !elements.approveInterpreterProposal.checked) {
    throw new Error("Review and approve the exact proposal before accepting it.");
  }
  const result = await withBusy(
    "Accepting interpreter proposal",
    "Revalidating the checksum-bound source and recording provider provenance without running analysis…",
    () => engine("accept_interpreter_proposal", {
      proposal: reviewed.proposal,
      expected_proposal_sha256: reviewed.proposal.proposal_sha256,
      current_model_source: elements.source.value,
    }),
  );
  const previousParameterValues = state.model ? parameterValues() : {};
  const retainedParameterValues = Object.fromEntries(
    result.model.structure.parameters
      .filter((parameter) => {
        const value = previousParameterValues[parameter.name];
        return Number.isFinite(value) && value >= parameter.lower && value <= parameter.upper;
      })
      .map((parameter) => [parameter.name, previousParameterValues[parameter.name]]),
  );
  const lineage = [...state.interpreterAcceptances, result.acceptance];
  const sourceName = state.sourceName;
  state.openedDocument = null;
  state.experimentInspection = null;
  elements.openedExperiment.classList.add("hidden");
  setSource(result.source, sourceName, null);
  state.interpreterAcceptances = lineage;
  applyModel(result.model, {
    initialValues: retainedParameterValues,
    preserveSettings: true,
  });
  renderProvenance(result.model.provenance || []);
  clearInterpreterProposal();
  clearInterpreterClarification(true);
  setStatus("Interpreter proposal accepted and validated. No analysis was run.", `Acceptance ${shortHash(result.acceptance.acceptance_sha256)}`);
  toast("Validated interpreter proposal accepted.");
}

async function cancelInterpreterOperation() {
  const result = await native("interpreter_cancel");
  if (result.cancelled) setStatus("Cancelling the local interpreter operation…");
}

async function loadExample() {
  const result = await withBusy("Loading example", "Opening the bundled validated quadratic model…", () => engine("example_model"));
  state.openedDocument = null;
  state.experimentInspection = null;
  elements.openedExperiment.classList.add("hidden");
  setSource(result.source, "quadratic.yaml");
  applyModel(result.model);
  await runAnalysis();
}

async function initialize() {
  renderCommittedBoundary();
  if (!invoke) {
    elements.engineDot.classList.add("error");
    elements.engineLabel.textContent = "Native shell unavailable";
    setStatus("Open this project with the Tauri development command.");
    return;
  }
  try {
    const health = await engine("health");
    elements.engineDot.classList.add("ready");
    elements.engineLabel.textContent = `Scientific engine ${health.version}`;
    elements.statusContext.textContent = `Desktop v${health.version}`;
    refreshInterpreterStatus().catch(() => {});
    const opened = await drainPendingDocuments();
    if (!opened) {
      await loadExample();
    }
  } catch (error) {
    elements.engineDot.classList.add("error");
    elements.engineLabel.textContent = "Scientific engine unavailable";
  }
}

async function handleAssociatedDocument(document) {
  if (/\.mlab$|\.json$/i.test(document.name)) {
    await inspectExperiment({ name: document.name, path: document.path, dataBase64: document.dataBase64 });
  } else {
    const source = base64ToText(document.dataBase64);
    setSource(source, document.name, document.path);
    await withBusy("Validating model", "Opening the associated model source…", () => validateCurrentModel());
  }
}

async function drainPendingDocuments() {
  let opened = false;
  while (true) {
    const document = await native("startup_document");
    if (!document) return opened;
    opened = true;
    await handleAssociatedDocument(document);
  }
}

async function boot() {
  if (window.__TAURI__?.event?.listen) {
    await window.__TAURI__.event.listen("external-document", () => {
      drainPendingDocuments().catch(() => {});
    });
  }
  await initialize();
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));

elements.source.addEventListener("input", () => {
  clearInterpreterProposal();
  clearInterpreterClarification(true);
  clearInterpreterLineage();
  state.source = elements.source.value;
  state.openedDocument = null;
  state.experimentInspection = null;
  elements.openedExperiment.classList.add("hidden");
  updateSourcePosition();
  invalidateModel();
});

byId("validate-button").addEventListener("click", () => withBusy("Validating model", "Parsing source and constructing canonical Model IR…", () => validateCurrentModel()).catch(() => {
  elements.source.classList.add("invalid");
  setChip(elements.validationChip, "Invalid", "error");
}));
byId("analyse-button").addEventListener("click", () => runPrimaryAnalysis().catch(() => {}));
elements.capabilitySelect.addEventListener("change", () => updateCapabilitySelection());
byId("capability-run-button").addEventListener("click", () => runSelectedCapability().catch(() => {}));
byId("add-capability-run-button").addEventListener("click", () => {
  try { addSelectedCapabilityToPlan(); } catch (error) { toast(error.message, "error"); }
});
byId("clear-capability-plan-button").addEventListener("click", () => {
  if (state.capabilityPlan.length) state.inputRevision += 1;
  clearCapabilityPlan();
  clearPreparedState("The author review was cleared because the generic run plan changed.");
  renderCommittedBoundary();
});
byId("run-sweep-button").addEventListener("click", (event) => { event.preventDefault(); runSweep().catch(() => {}); });
elements.sweepForm.addEventListener("submit", (event) => { event.preventDefault(); runSweep().catch(() => {}); });
byId("new-model-button").addEventListener("click", () => {
  state.openedDocument = null;
  state.experimentInspection = null;
  elements.openedExperiment.classList.add("hidden");
  setSource("name: Untitled model\n\nvariables:\n  x:\n    domain: [-10, 10]\n\nfunctions:\n  y: x**2\n", "Untitled model");
  setStatus("New model source created.");
});
byId("open-model-button").addEventListener("click", () => openModel().catch(() => {}));
byId("open-experiment-button").addEventListener("click", () => openExperiment().catch(() => {}));
byId("save-model-button").addEventListener("click", () => saveModel().catch(() => {}));
byId("load-example-button").addEventListener("click", () => loadExample().catch(() => {}));
byId("prepare-experiment-button").addEventListener("click", () => prepareExperiment().catch(() => {}));
byId("save-draft-button").addEventListener("click", () => {
  if (!state.prepared) return;
  saveEncoded("experiment", defaultExperimentName(state.model?.name, "-draft.mlab"), state.prepared.draft_bundle_base64).catch(() => {});
});
byId("save-state-button").addEventListener("click", () => {
  if (!state.prepared) return;
  saveEncoded("experiment", defaultExperimentName(state.model?.name, ".json"), textToBase64(state.prepared.state_json)).catch(() => {});
});
byId("commit-state-button").addEventListener("click", () => commitPreparedExperiment().catch(() => {}));
byId("open-commit-button").addEventListener("click", () => openCommittedReceipt().catch(() => {}));
byId("save-commit-button").addEventListener("click", () => saveCommittedReceipt().catch(() => {}));
elements.approveReview.addEventListener("change", () => {
  byId("publish-experiment-button").disabled = !(state.prepared && elements.approveReview.checked);
});
byId("publish-experiment-button").addEventListener("click", () => finalizeExperiment().catch(() => {}));
byId("reset-parameters-button").addEventListener("click", () => {
  if (!state.model) return;
  renderParameterControls(state.model);
  invalidateComputedResults(
    "parameter-values",
    "The author review was cleared because parameter values changed.",
  );
});
elements.visualisationSelect.addEventListener("change", () => {
  invalidateComputedResults(
    "visualisation-select",
    "The author review was cleared because the selected visualisation changed.",
  );
  if (state.model) runAnalysis().catch(() => {});
});
elements.sweepParameter.addEventListener("change", () => {
  const parameter = state.model?.structure.parameters.find((item) => item.name === elements.sweepParameter.value);
  if (parameter) { elements.sweepStart.value = parameter.lower; elements.sweepEnd.value = parameter.upper; }
  invalidateComputedResults("sweep-parameter", "The author review was cleared because sweep settings changed.");
});

[
  "points-1d", "points-2d", "samples-1d", "seeds-2d", "root-tolerance",
].forEach((id) => byId(id).addEventListener("input", () => invalidateComputedResults(
  id,
  "The author review was cleared because numerical settings changed.",
)));

["relative-tolerance", "absolute-tolerance"].forEach((id) => byId(id).addEventListener("input", () => invalidateComputedResults(
  id,
  "The author review was cleared because reproduction tolerances changed.",
)));

["sweep-start", "sweep-end", "sweep-steps"].forEach((id) => byId(id).addEventListener("input", () => invalidateComputedResults(
  id,
  "The author review was cleared because sweep settings changed.",
)));

elements.sweepVisualisation.addEventListener("change", () => invalidateComputedResults(
  "sweep-visualisation",
  "The author review was cleared because sweep settings changed.",
));

byId("include-sweep").addEventListener("change", () => {
  state.inputRevision += 1;
  clearPreparedState("The author review was cleared because experiment analyses changed.");
  renderCommittedBoundary();
});

elements.interpreterInstruction.addEventListener("input", () => {
  if (state.interpreterConversation
    && state.interpreterConversation.instruction !== elements.interpreterInstruction.value.trim()) {
    clearInterpreterClarification(true);
  }
  syncInterpreterControls();
});
elements.interpreterClarificationAnswer.addEventListener("input", syncInterpreterControls);
byId("interpreter-refresh-button").addEventListener("click", () => refreshInterpreterStatus().catch((error) => {
  toast(error instanceof Error ? error.message : String(error), "error");
}));
byId("interpreter-pull-button").addEventListener("click", () => pullInterpreterModel().catch(() => {}));
elements.interpreterAttachButton.addEventListener("click", () => attachInterpreterFile().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(message);
  toast(message, "error");
}));
byId("interpreter-request-button").addEventListener("click", () => requestInterpreterProposal().catch(() => {}));
byId("interpreter-clarification-continue").addEventListener("click", () => continueInterpreterClarification().catch((error) => {
  toast(error instanceof Error ? error.message : String(error), "error");
}));
byId("interpreter-clarification-cancel").addEventListener("click", () => {
  clearInterpreterClarification(true);
  syncInterpreterControls();
  setStatus("Interpreter clarification dialogue cancelled. The model source was not changed.");
});
byId("interpreter-cancel-button").addEventListener("click", () => cancelInterpreterOperation().catch(() => {}));
elements.approveInterpreterProposal.addEventListener("change", () => {
  byId("interpreter-accept-button").disabled = !(state.interpreterProposal && elements.approveInterpreterProposal.checked);
});
byId("interpreter-accept-button").addEventListener("click", () => acceptInterpreterProposal().catch(() => {}));
byId("interpreter-reject-button").addEventListener("click", () => {
  clearInterpreterProposal();
  clearInterpreterClarification(true);
  setStatus("Interpreter proposal rejected. The model source was not changed.");
});

window.addEventListener("resize", () => {
  resizePlot(elements.modelPlot);
  resizePlot(elements.sweepPlot);
  resizePlot(elements.capabilityPlot);
  const reproductionPlot = byId("reproduction-plot");
  if (reproductionPlot) resizePlot(reproductionPlot);
});

boot();
