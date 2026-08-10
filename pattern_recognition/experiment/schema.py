"""Pydantic experiment configuration models."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SplitConfig(BaseModel):
    """Epoch-level split parameters (Samara; never subject policy).

    Shared with the speller benchmark so binary ``config.json`` and speller
    configs can carry identical ``split`` blocks.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    epoch_holdout: float = 0.3
    stratify: bool = True
    val_fraction: float = 0.2


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
    split: Optional[SplitConfig] = None
    subject_mode: Literal["within_subject", "cross_subject", "mixed_pool"] = (
        "within_subject"
    )
    test_subjects: Optional[list[str]] = None

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
