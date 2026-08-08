"""Experiment package public API."""

from pattern_recognition.experiment.runner import run_experiment
from pattern_recognition.experiment.schema import ExperimentConfig

__all__ = ["ExperimentConfig", "run_experiment"]
