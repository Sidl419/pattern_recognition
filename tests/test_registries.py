# tests/test_registries.py
import pytest
from pattern_recognition.data.pipelines.registry import get_pipeline, list_pipelines
from pattern_recognition.models.registry import get_model, list_models


def test_synthetic_pipeline_registered():
    assert "SyntheticBinary" in list_pipelines()
    pipe = get_pipeline("SyntheticBinary")(n_train=32, n_val=16, n_channels=1, n_times=64)
    bundle = pipe.build()
    assert bundle.train is not None and bundle.val is not None


def test_real_pipelines_registered():
    names = list_pipelines()
    assert "SamaraWithinSubjectAverage" in names
    assert "SamaraTimeShift" in names
    assert "BCI3PzEpochAverage" in names


def test_unknown_pipeline_lists_options():
    with pytest.raises(KeyError, match="SyntheticBinary"):
        get_pipeline("DoesNotExist")


def test_eegnet_and_basecnn_registered():
    assert "EEGNet" in list_models()
    assert "BaseCNN" in list_models()
    m = get_model("EEGNet")(input_feat_dim=64, n_channels=1)
    assert m is not None


def test_unknown_model_lists_options():
    with pytest.raises(KeyError, match="EEGNet"):
        get_model("Nope")
