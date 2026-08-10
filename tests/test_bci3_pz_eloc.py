import pytest

from pattern_recognition.data.pipelines.bci3_pz import BCI3PzEpochAverage


def test_bci3_eloc_errors(tmp_path):
    train_mat = tmp_path / "Subject_A_Train.mat"
    train_mat.touch()
    with pytest.raises(ValueError, match="eloc_path is required"):
        BCI3PzEpochAverage(train_mat=str(train_mat)).build()
    with pytest.raises(FileNotFoundError, match="Electrode montage not found"):
        BCI3PzEpochAverage(
            train_mat=str(train_mat),
            eloc_path=str(tmp_path / "eloc64.loc"),
        ).build()
