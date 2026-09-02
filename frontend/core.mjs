export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function base64ToBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

export function bytesToBase64(bytes) {
  const chunkSize = 0x8000;
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, Math.min(offset + chunkSize, bytes.length));
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

export function base64ToText(value) {
  return new TextDecoder("utf-8", { fatal: true }).decode(base64ToBytes(value));
}

export function textToBase64(value) {
  return bytesToBase64(new TextEncoder().encode(value));
}

export function shortHash(value, leading = 10, trailing = 6) {
  const text = String(value ?? "");
  if (text.length <= leading + trailing + 1) return text || "—";
  return `${text.slice(0, leading)}…${text.slice(-trailing)}`;
}

export function reportTone(status) {
  if (status === "EXACT REPRODUCTION") return "exact";
  if (status === "NUMERICALLY REPRODUCED") return "numerical";
  if (status === "STATISTICALLY REPRODUCED") return "statistical";
  if (status === "PARTIALLY REPRODUCED") return "partial";
  return "failed";
}

export function statusTone(status) {
  const normalized = String(status ?? "").toLowerCase();
  if (normalized.includes("exact") || normalized.includes("statistically reproduced") || normalized === "complete" || normalized === "ok") return "success";
  if (normalized.includes("partial") || normalized.includes("warning") || normalized.includes("different")) return "warning";
  if (normalized.includes("unable") || normalized.includes("not reproduced") || normalized.includes("error") || normalized.includes("failed")) return "error";
  return "neutral";
}

const ANALYSIS_INPUTS = new Set([
  "parameter-values",
  "visualisation-select",
  "points-1d",
  "points-2d",
  "samples-1d",
  "seeds-2d",
  "root-tolerance",
]);

const SWEEP_INPUTS = new Set([
  "sweep-parameter",
  "sweep-start",
  "sweep-end",
  "sweep-steps",
  "sweep-visualisation",
]);

const REPRODUCTION_INPUTS = new Set([
  "relative-tolerance",
  "absolute-tolerance",
]);

export function resultInvalidationForInput(identifier) {
  if (ANALYSIS_INPUTS.has(identifier)) {
    return { analysis: true, sweep: true, reproduction: true };
  }
  if (SWEEP_INPUTS.has(identifier)) {
    return { analysis: false, sweep: true, reproduction: false };
  }
  if (REPRODUCTION_INPUTS.has(identifier)) {
    return { analysis: false, sweep: false, reproduction: true };
  }
  return { analysis: false, sweep: false, reproduction: false };
}

export function liveCommitRelationship(committedInputRevision, currentRevision) {
  if (!Number.isInteger(committedInputRevision)) return "uncommitted";
  if (!Number.isInteger(currentRevision)) return "diverged";
  return committedInputRevision === currentRevision ? "matches" : "diverged";
}

export function shouldCommitResult(requestRevision, currentRevision) {
  return Number.isInteger(requestRevision)
    && Number.isInteger(currentRevision)
    && requestRevision === currentRevision;
}

export function tableHtml(columns, rows, emptyMessage = "No records.") {
  if (!Array.isArray(rows) || rows.length === 0) {
    return `<div class="notice neutral">${escapeHtml(emptyMessage)}</div>`;
  }
  const safeColumns = Array.isArray(columns) && columns.length ? columns : Object.keys(rows[0]);
  const heading = safeColumns.map((column) => `<th scope="col">${escapeHtml(column)}</th>`).join("");
  const body = rows.map((row) => {
    const cells = safeColumns.map((column) => {
      const value = row?.[column];
      const display = Array.isArray(value) ? value.join(", ") : value ?? "—";
      return `<td>${escapeHtml(display)}</td>`;
    }).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  return `<div class="data-table-wrap"><table class="data-table"><thead><tr>${heading}</tr></thead><tbody>${body}</tbody></table></div>`;
}

export function metricHtml(label, value) {
  return `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? "—")}</strong></div>`;
}

export function defaultExperimentName(modelName, suffix = ".mlab") {
  const stem = String(modelName || "experiment")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "") || "experiment";
  return `${stem}${suffix}`;
}

export function scientificDisplay(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  if (!Number.isFinite(Number(value))) return String(value);
  const numeric = Number(value);
  if (numeric === 0) return "0";
  if (Math.abs(numeric) >= 1e5 || Math.abs(numeric) < 1e-4) return numeric.toExponential(6);
  return numeric.toPrecision(8).replace(/\.?0+$/, "");
}

export function validateNumericalSettings(settings) {
  const errors = [];
  if (!Number.isInteger(settings.points_1d) || settings.points_1d < 2 || settings.points_1d > 200000) errors.push("1D evaluation points must be between 2 and 200000.");
  if (!Number.isInteger(settings.points_per_axis_2d) || settings.points_per_axis_2d < 2 || settings.points_per_axis_2d > 1000) errors.push("2D points per axis must be between 2 and 1000.");
  if (!Number.isInteger(settings.samples_1d) || settings.samples_1d < 10 || settings.samples_1d > 200001) errors.push("1D stationary samples must be between 10 and 200001.");
  if (!Number.isInteger(settings.seeds_per_axis_2d) || settings.seeds_per_axis_2d < 3 || settings.seeds_per_axis_2d > 101) errors.push("2D seeds per axis must be between 3 and 101.");
  if (!(settings.root_tolerance > 0) || !Number.isFinite(settings.root_tolerance)) errors.push("Root tolerance must be positive and finite.");
  if (!(settings.relative_tolerance > 0) || settings.relative_tolerance > 1e-4 || !Number.isFinite(settings.relative_tolerance)) errors.push("rtol must be positive, finite, and no greater than 1e-4.");
  if (!(settings.absolute_tolerance > 0) || settings.absolute_tolerance > 1e-6 || !Number.isFinite(settings.absolute_tolerance)) errors.push("atol must be positive, finite, and no greater than 1e-6.");
  return errors;
}
