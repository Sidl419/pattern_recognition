"""Data pipelines package — import modules for side-effect registration."""

from pattern_recognition.data.pipelines.base import DataPipeline, DatasetBundle
from pattern_recognition.data.pipelines.registry import (
    get_pipeline,
    list_pipelines,
    register_pipeline,
)
from pattern_recognition.data.pipelines import synthetic as _synthetic  # noqa: F401

__all__ = [
    "DataPipeline",
    "DatasetBundle",
    "get_pipeline",
    "list_pipelines",
    "register_pipeline",
]
