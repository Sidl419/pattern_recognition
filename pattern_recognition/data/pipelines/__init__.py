"""Data pipelines package — import modules for side-effect registration."""

from pattern_recognition.data.pipelines import bci3_pz as _bci3_pz  # noqa: F401
from pattern_recognition.data.pipelines import (
    samara_average as _samara_average,  # noqa: F401
)
from pattern_recognition.data.pipelines import (
    samara_time_shift as _samara_time_shift,  # noqa: F401
)
from pattern_recognition.data.pipelines import synthetic as _synthetic  # noqa: F401
from pattern_recognition.data.pipelines import (
    synthetic_selection as _synthetic_selection,  # noqa: F401
)
from pattern_recognition.data.pipelines.base import DataPipeline, DatasetBundle
from pattern_recognition.data.pipelines.registry import (
    get_pipeline,
    list_pipelines,
    register_pipeline,
)

__all__ = [
    "DataPipeline",
    "DatasetBundle",
    "get_pipeline",
    "list_pipelines",
    "register_pipeline",
]
