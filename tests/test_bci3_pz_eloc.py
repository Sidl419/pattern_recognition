import pytest

from pattern_recognition.data.pipelines.bci3_pz import BCI3PzEpochAverage


def test_missing_eloc_path_raises_before_load(tmp_path):
    train_mat = tmp_path / "Subject_A_Train.mat"
    train_mat.touch()
    pipe = BCI3PzEpochAverage(train_mat=str(train_mat))
    with pytest.raises(ValueError, match="eloc_path is required"):
        pipe.build()


def test_missing_eloc_file_raises(tmp_path):
    train_mat = tmp_path / "Subject_A_Train.mat"
    train_mat.touch()
    pipe = BCI3PzEpochAverage(
        train_mat=str(train_mat),
        eloc_path=str(tmp_path / "eloc64.loc"),
    )
    with pytest.raises(FileNotFoundError, match="Electrode montage not found"):
        pipe.build()
