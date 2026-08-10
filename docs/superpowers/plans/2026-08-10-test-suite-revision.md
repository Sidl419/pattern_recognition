# Test Suite Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate and quality-revise the whole `tests/` tree (~95 → ~70–85 tests) per `docs/superpowers/specs/2026-08-10-test-suite-revision-design.md`.

**Architecture:** Shared `tests/speller/conftest.py` helpers; merge plot/meta modules; dedupe split/holdout and thin schema/CLI; no production changes unless tests expose bugs.

**Tech Stack:** pytest, matplotlib Agg, existing `pattern_recognition` package APIs.

## Global Constraints

- Absolute imports only (`from pattern_recognition...`).
- Do not commit unless the user explicitly asks.
- Keep review-fix guards listed in the design spec.
- Optional real-mat tests stay `@pytest.mark.skipif`.
- Target ~70–85 collected tests; full suite green.

---

### Task 1: Shared fixtures

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/speller/conftest.py`

**Produces:**
- `stub_binary_run(run_dir: Path, *, n_average: int = 1, device: str = "cpu", with_run_meta: bool = True, split: dict | None = ..., subject_mode: str = "within_subject", **config_extra) -> None`
- `samara_smoke_config(run_dir: Path | str, *, tag: str = "smoke", plots: bool = False, use_synthetic: bool = True, **overrides) -> dict`
- `write_speller_artifacts(speller_dir: Path, *, label_source: str = "simulated", early_stop: bool = False, n_selections: int = 2) -> None`
- Root `pytest_configure` or import-time `matplotlib.use("Agg")` in `tests/conftest.py`

- [ ] **Step 1:** Implement helpers matching current stub payloads (shared `split` block, `n_average`, optional `run_meta` with `device_*`).
- [ ] **Step 2:** Migrate `tests/speller/test_benchmark_smoke.py` to use helpers; remove local `_stub_binary_run` / `_smoke_config`.
- [ ] **Step 3:** Run `poetry run pytest tests/speller/test_benchmark_smoke.py -q` — expect PASS.

---

### Task 2: Merge plots + meta/load; delete obsolete modules

**Files:**
- Modify: `tests/speller/test_plots.py`
- Modify: `tests/speller/test_benchmark_smoke.py` (add meta device assertion into smoke)
- Modify: `tests/speller/test_leakage_and_itr.py` (use `stub_binary_run`)
- Delete: `tests/speller/test_extra_plots.py`
- Delete: `tests/speller/test_meta_and_tag_load.py`

- [ ] **Step 1:** Move extra plot tests + `write_speller_artifacts` usage into `test_plots.py`; move `load_speller_tag` / collision resolve tests into `test_plots.py`.
- [ ] **Step 2:** Assert `device_requested` / `device_resolved` in smoke meta test (or one dedicated test in smoke file).
- [ ] **Step 3:** Delete obsolete modules; run `poetry run pytest tests/speller/ -q` — expect PASS.

---

### Task 3: Dedupe split/holdout; tidy schema + CLI + package tests

**Files:**
- Modify: `tests/speller/test_simulate.py` — keep packing/seed/no-reuse; remove holdout cases covered by `tests/test_split_config.py` (keep one authoritative `test_shared_split_train_eval_indices_disjoint` in `test_split_config.py` or simulate — prefer `test_split_config.py` + packing-only in simulate).
- Modify: `tests/speller/test_schema.py` — parametrize near-duplicate rejects where safe.
- Modify: `tests/speller/test_cli.py` — single help smoke if both only check help exit/usage.
- Light pass: ensure no leftover private stubs; package-level files remain focused.

- [ ] **Step 1:** Relocate/dedupe disjoint-index test; trim simulate holdout duplicate.
- [ ] **Step 2:** Parametrize schema rejects; collapse CLI help.
- [ ] **Step 3:** Run `poetry run pytest tests/ -q --collect-only` and full `poetry run pytest tests/ -q`. Expect green and count in ~70–85.

---

### Task 4: Verify success criteria

- [ ] Confirm deleted files gone.
- [ ] Confirm no duplicate `_stub_binary_run` definitions under `tests/`.
- [ ] Confirm must-retain tests still collected (split gate, n_average, train pool, per-subject ITR, BCI3 scaler, flash scorer, skipif real mats).
