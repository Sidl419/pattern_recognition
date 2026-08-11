"""BCI3 / Samara selection-packet pipelines (monkeypatch when mats absent)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pattern_recognition.data.pipelines import get_pipeline
from pattern_recognition.speller.packing import collate_selection_packets
from pattern_recognition.speller.types import Selection
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
BCI3_TRAIN = ROOT / "matrix_dataset" / "Subject_A_Train.mat"
BCI3_TEST = ROOT / "matrix_dataset" / "Subject_A_Test.mat"
BCI3_ELOC = ROOT / "matrix_dataset" / "eloc64.loc"
SAMARA_DIR = ROOT / "Samara_data"
if not SAMARA_DIR.is_dir():
    SAMARA_DIR = ROOT / "raw_data"


def _fake_bci3_selection(char: str = "A", n_times: int = 8) -> Selection:
    codes = np.arange(1, 13, dtype=np.int64)
    flashes = np.zeros((12, 1, n_times), dtype=np.float32)
    return Selection(
        flashes=flashes,
        stimulus_ids=codes,
        target_char=char,
        repeat_index=np.zeros(12, dtype=np.int64),
        meta={"label_source": "ground_truth"},
    )


@pytest.mark.parametrize(
    ("pipeline", "ok_protocol", "bad_protocol", "kwargs_factory"),
    [
        (
            "BCI3SelectionPackets",
            "bci3_rowcol",
            "samara_single_flash_sim",
            lambda p: {
                "train_mat": str(p / "Subject_A_Train.mat"),
                "eloc_path": str(p / "eloc64.loc"),
            },
        ),
        (
            "SamaraSelectionPackets",
            "samara_single_flash_sim",
            "bci3_rowcol",
            lambda p: {"path": str(p / "Samara_data")},
        ),
    ],
)
def test_selection_packet_pipeline_protocol_validation(
    tmp_path, pipeline, ok_protocol, bad_protocol, kwargs_factory
):
    """Matching protocol is stored; mismatched protocol is rejected."""
    if pipeline.startswith("BCI3"):
        (tmp_path / "Subject_A_Train.mat").write_bytes(b"x")
        (tmp_path / "eloc64.loc").write_text("dummy")
    else:
        (tmp_path / "Samara_data").mkdir()
    kwargs = kwargs_factory(tmp_path)
    pipe = get_pipeline(pipeline)(**kwargs, protocol=ok_protocol)
    assert pipe.protocol == ok_protocol
    with pytest.raises(ValueError, match="protocol"):
        get_pipeline(pipeline)(**kwargs, protocol=bad_protocol)


def test_bci3_selection_packets_monkeypatched(monkeypatch, tmp_path):
    train_mat = tmp_path / "Subject_A_Train.mat"
    test_mat = tmp_path / "Subject_A_Test.mat"
    eloc = tmp_path / "eloc64.loc"
    train_mat.write_bytes(b"x")
    test_mat.write_bytes(b"x")
    eloc.write_text("dummy")

    def fake_build(mat_path, **kwargs):
        path = str(mat_path)
        if "Train" in path:
            return [_fake_bci3_selection("A"), _fake_bci3_selection("B")]
        return [_fake_bci3_selection("C")]

    monkeypatch.setattr(
        "pattern_recognition.speller.data_loading.build_bci3_selections_from_mat",
        fake_build,
    )

    pipe = get_pipeline("BCI3SelectionPackets")(
        train_mat=str(train_mat),
        test_mat=str(test_mat),
        eloc_path=str(eloc),
        subject="A",
        filter=False,
        max_chars=2,
        max_repetitions=1,
    )
    bundle = pipe.build()
    assert len(bundle.train) == 2
    assert len(bundle.val) == 1
    assert bundle.metadata["pipeline"] == "BCI3SelectionPackets"
    assert bundle.metadata["protocol"] == "bci3_rowcol"
    assert bundle.metadata["label_source"] == "ground_truth"

    loader = DataLoader(
        bundle.train, batch_size=2, collate_fn=collate_selection_packets
    )
    batch = next(iter(loader))
    assert batch["epochs"].ndim == 4
    assert "row_target" in batch and "column_target" in batch


def test_samara_selection_packets_monkeypatched(monkeypatch, tmp_path):
    data_dir = tmp_path / "Samara_data"
    data_dir.mkdir()
    # Enough epochs for phrase "JU", r_max=1 after holdout/val split.
    n_epochs = 200
    epochs = np.zeros((n_epochs, 250), dtype=np.float32)
    labels = np.array([i % 2 for i in range(n_epochs)], dtype=np.int64)

    def fake_load(path, **kwargs):
        return {"S0201": epochs}, {"S0201": labels}

    monkeypatch.setattr(
        "pattern_recognition.data.time_shift.load_p300_subjects",
        fake_load,
    )

    pipe = get_pipeline("SamaraSelectionPackets")(
        path=str(data_dir),
        phrase="JU",
        r_max=1,
        seed=0,
        epoch_holdout=0.3,
        val_fraction=0.25,
        stratify=True,
        channel_idx=1,
        epoch_len=250,
        subject="S0201",
    )
    bundle = pipe.build()
    assert len(bundle.train) == 2
    assert len(bundle.val) == 2
    assert bundle.metadata["pipeline"] == "SamaraSelectionPackets"
    assert bundle.metadata["protocol"] == "samara_single_flash_sim"
    assert bundle.metadata["label_source"] == "simulated"

    item = bundle.train[0]
    assert "epochs" in item and "stimulus_codes" in item
    assert "row_target" not in item
    assert item["character_target"].ndim == 0


@pytest.mark.skipif(
    not BCI3_TRAIN.is_file() or not BCI3_TEST.is_file() or not BCI3_ELOC.is_file(),
    reason="BCI3 mats missing",
)
def test_bci3_selection_packets_real_mats():
    pipe = get_pipeline("BCI3SelectionPackets")(
        train_mat=str(BCI3_TRAIN),
        test_mat=str(BCI3_TEST),
        eloc_path=str(BCI3_ELOC),
        subject="A",
        filter=False,
        max_chars=2,
        max_repetitions=1,
    )
    bundle = pipe.build()
    assert len(bundle.train) == 2
    assert len(bundle.val) == 2
    assert bundle.metadata["label_source"] == "ground_truth"


@pytest.mark.skipif(
    not SAMARA_DIR.is_dir() or not any(SAMARA_DIR.glob("S*-P300_classic.mat")),
    reason="Samara mats missing",
)
def test_samara_selection_packets_real_mats():
    subject = "S0201" if (SAMARA_DIR / "S0201-P300_classic.mat").is_file() else None
    pipe = get_pipeline("SamaraSelectionPackets")(
        path=str(SAMARA_DIR),
        phrase="JU",
        r_max=1,
        seed=0,
        epoch_holdout=0.3,
        val_fraction=0.2,
        subject=subject,
    )
    bundle = pipe.build()
    assert len(bundle.train) == 2
    assert len(bundle.val) == 2
    assert bundle.metadata["label_source"] == "simulated"
