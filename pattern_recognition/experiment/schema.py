"""Pydantic experiment configuration models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    pipeline: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    lr: float
    weight_decay: float
    batch_size: int
    num_epochs: int
    step_size: int
    gamma: float
    save_model: bool = True


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    device: str = "auto"
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    output_dir: str = "results/"

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        requested = value.strip().lower()
        if requested in {"auto", "cpu", "cuda"}:
            return requested
        if requested.startswith("cuda:"):
            suffix = requested.split(":", 1)[1]
            if not suffix.isdigit():
                raise ValueError(
                    f"Unknown device '{value}'. Expected auto|cpu|cuda|cuda:N."
                )
            return f"cuda:{int(suffix)}"
        raise ValueError(
            f"Unknown device '{value}'. Expected auto|cpu|cuda|cuda:N."
        )
