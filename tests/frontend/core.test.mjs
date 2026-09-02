import test from "node:test";
import assert from "node:assert/strict";

import {
  base64ToText,
  defaultExperimentName,
  escapeHtml,
  reportTone,
  resultInvalidationForInput,
  liveCommitRelationship,
  shouldCommitResult,
  shortHash,
  tableHtml,
  textToBase64,
  validateNumericalSettings,
} from "../../frontend/core.mjs";


test("UTF-8 documents round-trip through the native base64 payload", () => {
  const source = "name: λ model\nnotes: Δ = 2.7×10⁻¹²\n";
  assert.equal(base64ToText(textToBase64(source)), source);
});

test("dynamic scientific content is escaped before table rendering", () => {
  const rendered = tableHtml(["message"], [{ message: "<script>bad()</script>" }]);
  assert.doesNotMatch(rendered, /<script>/);
  assert.match(rendered, /&lt;script&gt;/);
  assert.equal(escapeHtml('a & "b"'), "a &amp; &quot;b&quot;");
});

test("reproduction statuses map to unambiguous report tones", () => {
  assert.equal(reportTone("EXACT REPRODUCTION"), "exact");
  assert.equal(reportTone("NUMERICALLY REPRODUCED"), "numerical");
  assert.equal(reportTone("STATISTICALLY REPRODUCED"), "statistical");
  assert.equal(reportTone("PARTIALLY REPRODUCED"), "partial");
  assert.equal(reportTone("UNABLE TO REPRODUCE"), "failed");
});

test("scientific settings are validated before crossing the process boundary", () => {
  const valid = {
    points_1d: 1000,
    points_per_axis_2d: 150,
    samples_1d: 2001,
    seeds_per_axis_2d: 11,
    root_tolerance: 1e-9,
    relative_tolerance: 1e-8,
    absolute_tolerance: 1e-11,
  };
  assert.deepEqual(validateNumericalSettings(valid), []);
  assert.match(validateNumericalSettings({ ...valid, points_per_axis_2d: 1001 })[0], /1000/);
});

test("portable filenames and compact hashes are deterministic", () => {
  assert.equal(defaultExperimentName("Reaction Model 7"), "reaction-model-7.mlab");
  assert.equal(shortHash("1234567890abcdefgh", 4, 3), "1234…fgh");
});

test("every experiment input invalidates the result class that depends on it", () => {
  assert.deepEqual(resultInvalidationForInput("parameter-values"), {
    analysis: true,
    sweep: true,
    reproduction: true,
  });
  assert.deepEqual(resultInvalidationForInput("points-1d"), {
    analysis: true,
    sweep: true,
    reproduction: true,
  });
  assert.deepEqual(resultInvalidationForInput("sweep-steps"), {
    analysis: false,
    sweep: true,
    reproduction: false,
  });
  assert.deepEqual(resultInvalidationForInput("relative-tolerance"), {
    analysis: false,
    sweep: false,
    reproduction: true,
  });
});

test("an asynchronous result is committed only for its original input revision", () => {
  assert.equal(shouldCommitResult(7, 7), true);
  assert.equal(shouldCommitResult(7, 8), false);
  assert.equal(shouldCommitResult(undefined, undefined), false);
});


test("live and committed experiment revisions remain distinct", () => {
  assert.equal(liveCommitRelationship(null, 4), "uncommitted");
  assert.equal(liveCommitRelationship(4, 4), "matches");
  assert.equal(liveCommitRelationship(4, 5), "diverged");
  assert.equal(liveCommitRelationship(4, 4), "matches");
});
