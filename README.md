# trace-anomaly-detector

**The early-warning layer every team running AI agents in production needs.**

Akash's agent-shipping story now has four legs: `mcp-tool-server` runs agents live against tools, the red-teaming harness + firewall grade and defend under attack, `tool-call-trace-replay` pins behavior in CI — and **this** one watches the **trace** — plain-English: the recorded step-by-step history of one agent run. stream and pages you *before* a bad session reaches a customer. Replay tells you what changed after the fact; the anomaly detector tells you something is wrong while the trace is still warm.

It learns what "normal" looks like from a corpus of benign traces (a **baseline** — plain-English: a statistical profile of usual behavior.), then scores every new session against it and emits structured findings with evidence: observed number vs. baseline number, so a human or a **CI** — plain-English: continuous integration, the automated checks that run on every code change. job can see exactly why it fired.

## Quickstart

```bash
# 1. Generate a fictional practice corpus (Helios Home store + Helios Cloud devops)
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from corpus import generate_normal_sessions, session_jsonl
import os
os.makedirs('traces', exist_ok=True)
for i, (meta, steps) in enumerate(generate_normal_sessions(700, seed=42)):
    open(f'traces/trace-{i:04d}.jsonl', 'w').write(session_jsonl(meta, steps))
EOF

# 2. Mine the baseline
python cli.py mine --traces traces --out baseline.json

# 3. Scan new traces (exit codes for CI: 0 clean / 1 warnings / 2 critical)
#    CI = continuous integration — plain-English: the automated checks that run on every code change.
python cli.py scan --baseline baseline.json --traces traces; echo "exit=$?"

# 4. Explain one trace in detail
python cli.py explain --baseline baseline.json --trace traces/trace-0007.jsonl

# 5. One-command end-to-end demo (asserts internally)
python cli.py demo

# 6. Tiny JSON API (API — plain-English: a way for programs to talk to this tool over HTTP.)
python cli.py serve --baseline baseline.json --port 8137
# GET  /baseline   -> the mined baseline
# POST /scan       -> {"trace_jsonl": "..."} or {"meta": {...}, "steps": [...]}
```

The trace format is the same JSONL the `tool-call-trace-replay` recorder writes (`meta` line + `step` lines: `seq`, `observation`, `tool`, `args`, `result`, `duration_ms`, …), plus one optional extension field: `tokens: {"input": n, "output": n}` per step. The replay debugger ignores fields it doesn't know, so traces stay compatible both ways.

## Architecture

```
corpus.py    seeded fictional trace generator (**seeded** — plain-English: driven by a fixed random seed, so the same seed always produces byte-identical output. deterministic: seed=42 -> byte-identical JSONL)
miner.py     learns the baseline: per-tool latency/token percentiles, per-session
             call-frequency stats, per-(tool, arg) value vocabularies, tool->tool
             bigram counts, session-length + error-rate stats  -> baseline.json
detector.py  scores a session against the baseline -> findings with evidence
cli.py       mine / scan / explain / demo / serve  (exit 0/1/2 for CI — see above)
demo.py      one-command end-to-end, asserts internally
evals/       golden evals: 35 labeled sessions, known anomaly labels
tests/       84 hermetic unit tests (**hermetic** — plain-English: fully self-contained; no network, no external services. stdlib unittest, no network)
```

**Stdlib only** (**stdlib** — plain-English: Python's built-in library; no third-party packages to install.). No third-party dependencies, no network calls in the pipeline (tests stub **sockets** — plain-English: the OS-level network connections programs open. to prove it), no API keys, no personal data — the corpus is 100% fictional Helios sample data.

## Anomaly types

| Type | What it catches | Rule (tunable in `detector.DEFAULT_CONFIG`) |
|---|---|---|
| `latency_spike` | a step took far longer than this tool usually does | `duration > p99` **and** `duration >= 2.5 × p50` |
| `token_blowup` | a step burned far more **tokens** — plain-English: the units of text an LLM reads/writes, the billing unit. than usual | `tokens > p99` **and** `tokens >= 3 × p50` |
| `tool_usage_drift` | a tool called far more times in one session than its norm | **z-score** — plain-English: how many standard deviations an observation sits from the average. `>= 3` of per-session count |
| `rare_arg_value` | an argument value the baseline rarely/never saw | never-seen, or relative frequency `< 1%` with `>= 200` baseline samples |
| `new_tool_never_seen` | a tool absent from the baseline entirely | always critical |
| `error_rate_surge` | a large share of the session's steps failed | `>= 30%` warn, `>= 50%` critical |
| `unusual_tool_sequence` | a tool-to-tool transition the baseline never recorded | unseen **bigram** — plain-English: a two-step sequence. after a well-observed tool |
| `session_length_anomaly` | a marathon session, or a session that died after one step | length z-score `>= 4`, or single step vs. `>= 3`-step norm |

A **percentile** — plain-English: the value below which a given percentage of observations fall (p99 = 99% of calls were faster). summarizes each tool's latency/token **distribution** — plain-English: the full spread of observed values. High-cardinality arguments (ids like `order_id`, unique per call) are detected at mine time and excluded from rarity checks — "never seen before" is meaningless when every value is unique.

Severe deviations escalate from `warning` to `critical`; the session **verdict** — plain-English: the overall judgment for one session: ok, suspicious, or critical. is the worst finding severity, and `scan` exits 0/1/2 accordingly.

## Evals

`evals/run_evals.py` builds a baseline from 700 benign sessions (seed=42), then scores 35 labeled sessions (seed=2026): 23 scripted anomalies across all 8 types + 12 benign controls. Results are committed in `evals/eval_report.json`:

- **Detection: 23/23 — per-type recall 1.0** (every scripted anomaly caught, including all 3 `session_length_anomaly` and both `new_tool_never_seen` cases)
- **False positives: 0/12 benign controls flagged (0.0)**
- `all_passed: true`
- Report md5 (**md5** — plain-English: a fingerprint hash of a file's bytes; identical files have identical hashes. `1c380faf88714f43ea9007822b2e3f35`, verified across 2+ runs): byte-identical across runs.

(**Recall** — plain-English: the fraction of real anomalies the detector actually caught. **False-positive rate** — plain-English: the fraction of clean sessions wrongly flagged.)

Note: hammering one tool also trips `session_length_anomaly` and `unusual_tool_sequence` — all three findings are true (the session *is* long and *does* contain unseen transitions), so **co-firing** — plain-English: multiple anomaly types firing on the same session. is by design, not noise.

## Dev loop — bugs the tests caught mid-build

1. **High-cardinality args poisoned rarity detection.** First eval run: 23/23 anomalies caught but benign FP was **0.4167** (5/12 clean sessions flagged). Root cause: `order_id`/`alert_id` are unique per call, so every value was "never seen". Fix: the miner flags arguments with >50 distinct values at >30% uniqueness as `high_cardinality`; the detector skips rarity checks for them and the baseline doesn't store their vocabularies. FP dropped to 0.0.
2. **Small-sample relative-frequency fluke.** `place_order.sku='HL-1006'` seen 1/101 times (0.99% < 1% threshold) fired as "rare" in the 300-session demo baseline — a sampling fluke, not a real signal. Fix: the relative-frequency branch now requires `rare_min_observations` (200) baseline samples; the "never seen" branch still fires regardless.
3. **Test-fixture cross-talk.** The synthetic "unknown tool c" in the sequence tests tripped `latency_spike` because its hand-built profile (p50=1ms) didn't match the 50ms default test steps. Fixture bug, not product bug — fixed by giving test fixtures realistic profiles. Lesson recorded: hand-built baselines must stay physically plausible.

## Lingo rule

This repo follows Akash's global lingo rule: real technical terms everywhere, each with a one-line plain-English explanation (`**term** — plain-English meaning.`) on every use.

## License

MIT — see [LICENSE](LICENSE). Copyright Akash Bhargava.
