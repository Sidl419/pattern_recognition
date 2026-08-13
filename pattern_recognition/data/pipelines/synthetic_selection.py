"""Synthetic selection-packet pipeline for sequence-model CI smoke tests."""

from __future__ import annotations

from pattern_recognition.data.datasets import SelectionPacketDataset
from pattern_recognition.data.pipelines.base import DatasetBundle
from pattern_recognition.data.pipelines.registry import register_pipeline
from pattern_recognition.selections.packing import PROTOCOLS, pack_selection
from pattern_recognition.selections.protocols import get_protocol


def _phrase_for_count(protocol_name: str, n: int, phrase: str | None) -> str:
    if phrase is not None:
        if len(phrase) < n:
            raise ValueError(f"phrase length {len(phrase)} < required selections {n}")
        return phrase[:n]
    chars = get_protocol(protocol_name).grid.chars
    return "".join(chars[i % len(chars)] for i in range(n))


@register_pipeline("SyntheticSelectionPackets")
class SyntheticSelectionPackets:
    """Build train/val selection packets via protocol synthetic schedules."""

    def __init__(
        self,
        protocol: str = "bci3_rowcol",
        n_train: int = 8,
        n_val: int = 4,
        n_channels: int = 1,
        n_times: int = 64,
        r_max: int = 2,
        seed: int = 0,
        phrase: str | None = None,
    ) -> None:
        if protocol not in PROTOCOLS:
            raise ValueError(
                f"Unknown protocol {protocol!r}; expected one of {sorted(PROTOCOLS)}"
            )
        self.protocol = protocol
        self.n_train = n_train
        self.n_val = n_val
        self.n_channels = n_channels
        self.n_times = n_times
        self.r_max = r_max
        self.seed = seed
        self.phrase = phrase

    def build(self) -> DatasetBundle:
        n_total = self.n_train + self.n_val
        phrase = _phrase_for_count(self.protocol, n_total, self.phrase)
        proto = get_protocol(self.protocol)
        selections = proto.build_synthetic_selections(
            phrase=phrase,
            r_max=self.r_max,
            seed=self.seed,
            flash_shape=(self.n_channels, self.n_times),
        )
        packets = [pack_selection(sel, protocol=self.protocol) for sel in selections]
        train = SelectionPacketDataset(packets[: self.n_train])
        val = SelectionPacketDataset(packets[self.n_train : self.n_train + self.n_val])
        return DatasetBundle(
            train=train,
            val=val,
            test=None,
            metadata={
                "pipeline": "SyntheticSelectionPackets",
                "protocol": self.protocol,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_channels": self.n_channels,
                "n_times": self.n_times,
                "r_max": self.r_max,
                "seed": self.seed,
                "phrase": phrase,
                "label_source": proto.label_source,
            },
        )
