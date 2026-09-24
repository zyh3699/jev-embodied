# Paired evaluation figure notes

This quantitative grid compares task outcomes, resource use, and request delay under one frozen protocol. It does not combine these measurements into an accuracy or intelligence score.

Coverage: partial; unfinished/error cases retained. Matched initial states: 4/4.

- Task success comes from the benchmark environment. Each replicate is one listed task/initial state/seed; the frozen manifest defines the split. For complete groups, bars/points show successes divided by planned episodes with a binomial Wilson 95% interval. For partial groups, hollow points show verified successes divided by planned episodes, a lower bound; no confidence interval or complete success-rate claim is shown. The JSON also retains the observed scored-only fraction, which is not used to hide missing cases.
- Token dots show episodes with complete provider-reported input or output usage; horizontal marks are medians. Missing usage is N/A. Partial reported totals and exact request coverage remain in the source data. Different providers may tokenize differently; these are usage counters, not equal units of computation or a billing estimate.
- API latency is measured client wall time per actual HTTP request, including error requests. p50 and p95 use linear interpolation of sorted measurements; n counts requests with measured latency. They are descriptive quantiles, not confidence intervals or server-only inference time.
- Wall time includes simulator setup and teardown. Simulator time, when available, is exported separately. Paired plots connect matching task/seed observations. Missing values are retained and labelled, not imputed.
- Request count is actual HTTP requests; the dashed cap is the model-attempt budget. These can differ after pre-request validation failures. Native steps use the benchmark controller frequency, preserved in metadata.
- No hypothesis test or multiple-comparison correction is applied. A small frozen subset is not a full official benchmark score or evidence of unseen-task generalization.

## Source and export audit

Manifest SHA-256: `2376f290d67f4464a3c15697c43123376242b365ce39e419f8091d02bee74be6`.
Report hashes use sorted-key UTF-8 JSON, matching the evaluator's manifest-hash convention; they are hashes of report contents, not original file whitespace. The renderer source hash is recorded separately.
`comparison.json` preserves source-code and report hashes, the protocol, paired coverage and all planned cases. `episodes.csv`, `api_calls.csv`, and `task_summary.csv` contain panel source measurements. Empty CSV fields are unavailable.
The Python renderer exports an approximately 183 × 194 mm figure, editable SVG text, TrueType PDF text, and a 600 dpi PNG. All panels use the same method color and marker. No raster image manipulation is involved.
The source preflight may warn that TIFF is absent: PNG is the requested raster preview and SVG/PDF provide the editable vector exports; this bundle is not a journal-specific submission package.
Rendering is an automated export, not visual approval. Inspect the exported figure at final size before publication.

- Jev: 4/4 verified successes; 0 missing, 0 unscored; 225 HTTP requests; latency n=225. Input usage reported for 224 requests; output usage for 224. Failure outcomes: {}.
- Chat: 3/4 verified successes; 0 missing, 0 unscored; 208 HTTP requests; latency n=208. Input usage reported for 206 requests; output usage for 206. Failure outcomes: {"runtime_error": 1}.
