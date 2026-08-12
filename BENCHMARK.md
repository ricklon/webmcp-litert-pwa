# Model benchmark

## Remembered v1 findings

The August 12, 2026 headed Chrome run used Apple Metal WebGPU and one run of
three managed scenarios. It is diagnostic evidence, not a statistically sound
ranking.

| Runtime | Availability | Accuracy | Mean workflow latency | Observed behavior |
| --- | --- | ---: | ---: | --- |
| Chrome built-in | Unavailable in Playwright's isolated profile | Not measured | Not measured | A personal Chrome profile may still have the Prompt API and Gemini Nano installed. |
| LiteRT-LM Gemma 4 E2B | Available | 1/3 | 3.01 s | Correctly decomposed the six-item trip, completed collateral work in the typo case, and emitted malformed JSON for newly finished work. |
| Bonsai 27B Q1_0 | Available | 0/3 | 25.10 s | Produced constrained JSON at roughly 9 tokens/s, asked unnecessary clarification twice, and collapsed six requested tasks into two. |

Both external models separately passed a simple real add-task proposal and
approval test. That proves the basic inference-to-tool path works, but not that
either model selects tools reliably on harder requests.

The local planner benchmark does **not** exercise native WebMCP. Local models
receive the same tool contracts but the app dispatches their approved calls
internally. Native WebMCP exposes those contracts through
`document.modelContext` to an external browser agent. The WebMCP registration
and write-review gateway have deterministic tests, but a real external agent's
tool discovery, selection, `pendingReview` handling, and post-approval behavior
remain unmeasured.

## V2 measurement contract

V2 should report evidence at three levels:

1. **Decision quality:** raw outcome, exact ordered tool calls and arguments,
   clarification behavior, schema/parse failures, and final-state correctness.
2. **Safety and workflow:** no writes before approval, no collateral writes,
   destructive-action confirmation, clarification follow-up, and recovery from
   invalid or ambiguous calls.
3. **Performance:** model activation time, per-request planner latency,
   end-to-end workflow latency, median and p95 latency, context use, reported
   throughput, and availability. Download, compile, warm inference, memory, and
   thermal behavior must not be conflated when the runtime cannot expose them
   separately.

The default suite repeats each case three times. Model order, run count, and
scenario set remain configurable. A model that is unavailable is recorded and
does not prevent the remaining models from running. Chrome uses the ignored,
dedicated `.playwright/chrome-ai-profile` persistent profile; the benchmark
refuses to automate a profile inside Chrome's normal user-data directory.

## Interpretation rules

- Final state alone is insufficient: redundant or unsafe calls can accidentally
  produce the expected state, so exact call traces are retained.
- A clarification is correct only when the scenario marks ambiguity as
  material. Unnecessary clarification and missed clarification are separate
  failure categories.
- Auto-approval is used only to score model decisions without manual input. It
  does not validate the normal proposal-review user experience.
- Report local planner results separately from native WebMCP results.
- Compare repeated results and failure distributions, not a single successful
  sample.
- Chrome results are comparable only when the benchmark profile actually has
  access to the Prompt API and its installed model.

## V2 one-run calibration

The August 12, 2026 calibration used the same seven cases once per model. These
numbers validate the harness and expose likely failure modes; use the default
three repetitions before making a release decision.

| Runtime | Strict score | Exact decisions | Clarification | Median workflow | p95 workflow |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chrome built-in | 5/7 | 70% | 0/1 correct | 1.82 s | 5.37 s |
| LiteRT-LM Gemma 4 E2B | 1/7 | 30% | 0/1 correct | 1.34 s | 2.87 s |
| Bonsai 27B Q1_0 | 5/7 | 80% | 1/1 correct | 27.94 s | 72.62 s |

Chrome passed typo recovery, collateral safety, newly completed work, six-task
decomposition, and clearing. It answered the daily plan's final list request
without calling `list_tasks`, and it chose `submit report` rather than asking
which ambiguous report to complete. Its first downloaded-model run also exposed
a destroyed base-session startup race; the app now rebuilds that session once,
and the clean calibration above was collected after the fix.

## Still needed after V2

- A completed three-run evaluation from the dedicated Chrome-capable profile.
- A real external WebMCP agent evaluation.
- Fresh-download versus cached-startup isolation.
- Time-to-first-token, peak memory, energy, and sustained thermal measurements.
- Larger adversarial, multilingual, long-context, timezone/DST, cancellation,
  offline, and concurrent-agent corpora.

## Current V2 extensions

The active suite now contains twelve cases. In addition to the original seven,
it checks unsupported capabilities, absent task targets, underspecified writes,
natural-language priority mapping, and prompt injection embedded in task data.
The list-tool step no longer exposes task state to the planner, so a direct
answer cannot receive strict tool-use credit.

Scoring now separates correct outcome, unordered tool selection, exact tool
order, absence of unexpected calls, clarification, and final state. Every model
trace retains structured-output diagnostics. LiteRT-LM reports whether its raw
response was valid, whether bounded syntax/alias recovery succeeded, and
whether one validation retry was required. A repaired plan is never executed
until the normal capability and argument validation passes.

Native WebMCP remains a distinct suite. Deterministic coverage now includes
registration schemas and lifecycle, immediate reads, write review, destructive
proposal cancellation, and the memory-initialization boundary. An opt-in real
Chrome check discovers all four tools and invokes `list_tasks` through
`document.modelContext`; it can be run with `npm run test:webmcp`.

### LiteRT recovery calibration

Two one-run calibrations of the expanded twelve-case suite demonstrate why the
default repeats matter. Both scored 6/12 strict passes, but they failed
different cases. Raw structured-output validity moved from 6/14 decisions
(43%) to 8/14 (57%). Bounded recovery salvaged seven outputs in the first run
and four in the second; one and two decisions respectively required the single
validation retry. Median workflow latency remained stable at roughly 1.34 s.

The adapter therefore removes a large serialization penalty without masking
the remaining semantic failures. Across the runs, LiteRT still selected
unrelated completions for unsupported or absent-target requests, inconsistently
handled ambiguity, sometimes performed collateral completion, and treated a
past-tense update as an answer instead of an add-then-complete action. Safety
decisions must be assessed over repeated trials and protected at the
application boundary rather than delegated solely to the model.

## Three-run baseline and focused variants

The complete three-repeat baseline contains 36 scenarios and 45 request steps
per model. The untouched pre-guardrail report is preserved as
`benchmark-results/baseline-v2-3run.json`.

| Runtime | Strict scenarios | Exact decisions | Safe tool steps | Clarification | Median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chrome built-in | 9/36 | 29% | 36% | 0/6 | 1.83 s | 4.84 s |
| LiteRT E2B + recovery | 24/36 | 67% | 73% | 3/6 | 1.34 s | 3.37 s |
| Bonsai 27B, reasoning off | 22/36 | 73% | 100% | 6/6, 14 asked | 25.91 s | 76.60 s |

Stable unsafe failures justified deterministic application guardrails. Chrome
and E2B missed ambiguous completion, absent completion targets, and unsupported
email requests in all three runs. The app now resolves explicit completion
targets independently, fails closed on absent or ambiguous targets, and blocks
unsupported imperative communication requests. Each trace retains the raw
model outcome and calls alongside the guarded decision and named intervention,
so application safety gains cannot inflate the model-only score.

The focused variants repeated ambiguity, newly finished work, six-way
decomposition, unsupported capability, and missing target three times:

- **LiteRT E4B:** 100% raw JSON validity, versus E2B's 45% full-baseline rate;
  raw exact decisions were 10/18 steps. It handled newly finished work and the
  unsupported request in all three raw runs, still missed ambiguity and absent
  targets in all three, and decomposed six tasks in only one of three runs.
  Median focused latency was 2.84 s. The guarded workflow passed 13/15
  scenarios. Report: `benchmark-results/litert-e4b-focused-3run.json`.
- **Bonsai reasoning on:** raw exact decisions fell to 7/18 steps. It asked 11
  clarifications for three required cases, and failed every newly-finished-work
  and six-way-decomposition run. Median focused latency remained 24.94 s, with
  54.53 s p95. Reasoning should remain off for this planner. Report:
  `benchmark-results/bonsai-thinking-focused-3run.json`.

The current recommendation is E4B when the additional latency and model size
are acceptable, backed by deterministic mutation guardrails. E2B remains the
fastest custom runtime but depends heavily on output recovery. Bonsai remains
the safest raw tool selector, but its latency and over-clarification make it a
poor default for interactive task planning.
