"""BCI speller benchmark: grids, decode, and evaluation helpers."""

from pattern_recognition.speller.benchmark import (
    FlashScorer,
    OracleFlashScorer,
    SklearnFlashScorer,
    load_flash_scorer_from_run,
    run_speller_benchmark,
)
from pattern_recognition.speller.schema import (
    OnlineConfig,
    SimulationConfig,
    SpellerBenchmarkConfig,
    SplitConfig,
)

__all__ = [
    "FlashScorer",
    "OnlineConfig",
    "OracleFlashScorer",
    "SklearnFlashScorer",
    "SimulationConfig",
    "SpellerBenchmarkConfig",
    "SplitConfig",
    "load_flash_scorer_from_run",
    "run_speller_benchmark",
]
