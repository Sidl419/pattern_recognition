# Test suite revision design

**Date:** 2026-08-10  
**Status:** Approved for planning (user approved §§1–3 in session)  
**Scope:** Cleanup + quality pass over the whole `tests/` tree after speller/review-fix growth (~95 tests).

**Related:** `docs/superpowers/specs/2026-08-09-bci-speller-benchmark-design.md`, review-fix test additions for shared `split`, flash-scorer gates, plots, BCI3 prep.

## Goals

1. **Cleanup** — one set of shared speller fixtures; no repeated `_stub_binary_run` / `_smoke_config` / `_write_speller_artifacts`.
2. **Quality** — merge overlapping cases; one sharp test per behavior; keep review-fix guards.
3. **Whole tree** — revise all of `tests/`, not only `tests/speller/`.
4. **Fewer tests OK** — merge/drop overlaps; target ~70–85 collected tests with equivalent risk coverage.

## Non-goals

- New product features or production behavior changes (unless a test exposes a clear bug).
- Rewriting into a full `unit/` vs `integration/` taxonomy.
- CI matrix / dependency changes.
- Clearing or migrating notebook tests.

## Approach

**Consolidate + merge overlaps** (chosen over fixture-only cleanup and over a full layer restructure).

## Shared fixtures (§1)

Add:

| Location | Contents |
|---|---|
| `tests/speller/helpers.py` | Canonical importable helpers: `stub_binary_run`, `samara_smoke_config`, `write_speller_artifacts` |
| `tests/conftest.py` | Force matplotlib Agg once for plot tests |
| `tests/speller/conftest.py` | Reserved for future fixtures |

Canonical helpers:

- `stub_binary_run(run_dir, *, n_average=1, device="cpu", with_run_meta=True, **extra_config)` — writes `config.json` (+ optional `run_meta.json`) suitable for Samara synthetic/oracle smokes and gate tests.
- `samara_smoke_config(run_dir, *, tag="smoke", plots=False, **overrides)` — valid `SpellerBenchmarkConfig` dict with matching `split`, `use_synthetic=True` by default for CI.
- `write_speller_artifacts(speller_dir, *, label_source="simulated", early_stop=False)` — metrics CSVs/JSON (+ early-stop artifacts when requested) for plot/load tests.

Remove local copies of those helpers from:

- `tests/speller/test_benchmark_smoke.py`
- `tests/speller/test_plots.py`
- `tests/speller/test_extra_plots.py` (file deleted after merge)
- `tests/speller/test_meta_and_tag_load.py` (file deleted after merge)
- `tests/speller/test_leakage_and_itr.py`

## File merges / drops (§2)

| Keep | Absorb / delete |
|---|---|
| `tests/speller/test_plots.py` | All of `test_extra_plots.py` (per-subject, early-stop hist, `save_speller_plots`, `compare_speller_runs`) |
| `tests/speller/test_benchmark_smoke.py` | Device fields in `meta.json` from `test_meta_and_tag_load.py` |
| `tests/speller/test_plots.py` (or smoke) | `load_speller_tag` / collision resolve tests from `test_meta_and_tag_load.py` — prefer next to other `load_speller_tag` tests in `test_plots.py` |
| `tests/test_split_config.py` | Authoritative home for three-way / holdout disjointness vs binary gates |
| `tests/speller/test_simulate.py` | Keep packing/seed/no-reuse only; drop holdout cases duplicated by `test_split_config` / `three_way` tests |

**Delete after merge:**

- `tests/speller/test_extra_plots.py`
- `tests/speller/test_meta_and_tag_load.py`

**Package-level tidy (light):**

- Leave small focused modules as-is (`test_device`, `test_imports`, registries, experiment schema, averaging/time-shift math).
- Parametrize near-duplicate rejects in `tests/speller/test_schema.py` where one rule has many inputs.
- Collapse CLI help smokes in `tests/speller/test_cli.py` to a single help coverage test if both only assert successful help exit / usage text.

**Must retain (do not merge away):**

- Shared-split / missing-split hard fail
- `n_average != 1` flash-scorer gate
- `allow_train_pool_eval` uses full pool
- Per-subject ITR from subject `char_acc`
- Train∩eval disjoint index check (single authoritative test)
- BCI3 Scaler prep / `scaled_with_mne_scaler`
- Flash-scorer checkpoint load
- Optional real-mat `@pytest.mark.skipif` tests

## Quality rules (§3)

- One behavior per test; name states the contract (`rejects_…`, `writes_…`, `prefers_…`).
- Assert outcomes (artifacts, metrics, exceptions), not incidental internals.
- Prefer real package APIs; mock only to isolate I/O when necessary.
- Absolute imports only (`from pattern_recognition...`).
- Keep optional real EEG tests skippable when mats are absent.

## Success criteria

- `poetry run pytest tests/ -q` passes.
- Collected test count in **~70–85** (down from 95) without losing review-fix guards listed above.
- No duplicated stub helpers across speller test modules.
- `test_extra_plots.py` and `test_meta_and_tag_load.py` removed.

## Out of scope

- New coverage for untested features unless needed to replace a dropped weak test.
- Production refactors beyond what broken tests force.
- Committing this work unless explicitly requested.

## Implementation notes (for plan)

1. Add `conftest` helpers first; migrate one consumer file; keep green.
2. Merge plot + meta/load tests; delete obsolete modules.
3. Deduplicate split/holdout; parametrize schema; thin CLI.
4. Full-suite pytest; adjust count vs criteria; stop when green and criteria met.
