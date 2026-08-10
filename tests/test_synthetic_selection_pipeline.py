from pattern_recognition.data.pipelines import get_pipeline
from pattern_recognition.speller.packing import collate_selection_packets
from torch.utils.data import DataLoader


def test_synthetic_selection_packets_bci3():
    pipe = get_pipeline("SyntheticSelectionPackets")(
        protocol="bci3_rowcol",
        n_train=4,
        n_val=2,
        n_channels=1,
        n_times=64,
        r_max=2,
        seed=0,
    )
    bundle = pipe.build()
    loader = DataLoader(
        bundle.train, batch_size=2, collate_fn=collate_selection_packets
    )
    batch = next(iter(loader))
    assert batch["epochs"].ndim == 4
    assert batch["flash_targets"].shape[:2] == batch["epochs"].shape[:2]
    assert "row_target" in batch and "column_target" in batch
