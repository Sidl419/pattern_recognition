"""Pydantic config models for the speller benchmark stage."""

from __future__ import annotations

from typing import Any, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pattern_recognition.speller.grids import SAMARA_GRID


class SplitConfig(BaseModel):
    """Epoch-level split parameters (Samara only; never subject policy)."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    epoch_holdout: float = 0.3
    stratify: bool = True
    val_fraction: float = 0.2


class SimulationConfig(BaseModel):
    """Samara simulated protocol: phrase packing into selections."""

    seed: Optional[int] = None
    phrase: str


class OnlineConfig(BaseModel):
    """Online decode / early-stop policy."""

    early_stop: bool = False
    margin_tau: Optional[float] = None


class SpellerBenchmarkConfig(BaseModel):
    """Speller benchmark stage configuration.

    Validation rules (keep this list in sync with validators):

    - ``split`` must not contain ``subject_mode`` or ``test_subjects`` — those
      fields are top-level only (enforced via ``SplitConfig`` ``extra=forbid``).
    - ``split`` may be ``null`` when ``protocol == bci3_rowcol`` (official
      train/test mats imply the flash split).
    - When ``protocol == bci3_rowcol`` and ``split`` is non-null,
      ``epoch_holdout`` in ``(0, 1]`` and ``stratify: true`` are forbidden.
    - When ``protocol == samara_single_flash_sim`` and
      ``allow_train_pool_eval`` is false, ``split`` is required with
      ``epoch_holdout`` in ``(0, 1)``.
    - When ``protocol == samara_single_flash_sim``, ``simulation`` is required
      with a non-empty ``phrase`` whose characters are all on the Samara grid;
      ``simulation.seed`` is optional (falls back to ``split.seed`` at runtime).
    - When ``protocol == bci3_rowcol``, ``simulation`` must be omitted or
      ``null``.
    - When ``subject_mode == cross_subject``, ``test_subjects`` must be
      non-empty.
    - When ``subject_mode == within_subject``, ``test_subjects`` must be
      ``null`` or empty (avoids silent ignore).
    - When ``online.early_stop`` is true, ``online.margin_tau`` must be set
      and ``> 0``.
    - ``model_mode == selection_classifier`` is allowed in v1; runtime checks
      that the ``run_dir`` checkpoint matches (not validated here).
    - Speller ``split`` and top-level subject policy must match the binary
      ``run_dir`` artifacts unless ``allow_split_mismatch`` (enforced at
      benchmark load, not here).
    - When ``allow_train_pool_eval`` is true, record in ``meta.json`` and
      warn (enforced at benchmark run, not here).
    - ``use_synthetic`` defaults to false: selections come from real EEG paths
      in the binary ``run_dir`` config and/or ``protocol_params``. Set
      ``use_synthetic=true`` only for CI / unit smoke (random synthetic flashes).
    """

    tag: str
    model_mode: Literal["flash_scorer", "selection_classifier"]
    protocol: Literal["samara_single_flash_sim", "bci3_rowcol"]
    subject_mode: Literal["within_subject", "cross_subject", "mixed_pool"] = (
        "within_subject"
    )
    test_subjects: Optional[list[str]] = None
    repetitions: list[int]
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    run_dir: str
    split: Optional[SplitConfig] = None
    simulation: Optional[SimulationConfig] = None
    protocol_params: dict[str, Any] = Field(default_factory=dict)
    use_synthetic: bool = False
    allow_split_mismatch: bool = False
    allow_train_pool_eval: bool = False
    plots: bool = True

    @model_validator(mode="after")
    def validate_coupling(self) -> Self:
        """Protocol-aware and subject-policy validators."""
        self._validate_online_early_stop()
        self._validate_subject_policy()
        if self.protocol == "bci3_rowcol":
            self._validate_bci3()
        elif self.protocol == "samara_single_flash_sim":
            self._validate_samara()
        return self

    def _validate_online_early_stop(self) -> None:
        # online.early_stop == true ⇒ margin_tau set and > 0
        if self.online.early_stop and (
            self.online.margin_tau is None or self.online.margin_tau <= 0
        ):
            raise ValueError(
                "online.early_stop requires online.margin_tau to be set and > 0"
            )

    def _validate_subject_policy(self) -> None:
        has_test_subjects = bool(self.test_subjects)
        if self.subject_mode == "within_subject" and has_test_subjects:
            raise ValueError(
                "test_subjects must be null or empty when subject_mode is "
                "within_subject"
            )
        if self.subject_mode == "cross_subject" and not has_test_subjects:
            raise ValueError(
                "test_subjects must be non-empty when subject_mode is "
                "cross_subject"
            )

    def _validate_bci3(self) -> None:
        if self.simulation is not None:
            raise ValueError(
                "simulation must be omitted or null for protocol bci3_rowcol"
            )
        if self.split is None:
            return
        if 0 < self.split.epoch_holdout <= 1:
            raise ValueError(
                "epoch_holdout is forbidden for protocol bci3_rowcol when split "
                "is set"
            )
        if self.split.stratify:
            raise ValueError(
                "stratify is forbidden for protocol bci3_rowcol when split is "
                "set"
            )

    def _validate_samara(self) -> None:
        if not self.allow_train_pool_eval:
            if self.split is None:
                raise ValueError(
                    "split is required for protocol samara_single_flash_sim "
                    "unless allow_train_pool_eval is true"
                )
            holdout = self.split.epoch_holdout
            if holdout is None or holdout <= 0 or holdout >= 1:
                raise ValueError(
                    "split.epoch_holdout must be in (0, 1) for protocol "
                    "samara_single_flash_sim unless allow_train_pool_eval is "
                    "true"
                )

        if self.simulation is None:
            raise ValueError(
                "simulation is required for protocol samara_single_flash_sim"
            )
        phrase = self.simulation.phrase
        if not phrase or not phrase.strip():
            raise ValueError(
                "simulation.phrase must be non-empty for protocol "
                "samara_single_flash_sim"
            )
        invalid = {c for c in phrase if c not in SAMARA_GRID.chars}
        if invalid:
            raise ValueError(
                f"simulation.phrase contains characters not on the Samara "
                f"grid: {sorted(invalid)}"
            )
