import pytest
from pattern_recognition.data.pipelines.registry import (
    get_pipeline,
    list_pipelines,
)
from pattern_recognition.models.registry import get_model, list_models


def test_expected_pipelines_and_models_registered():
    pipelines = list_pipelines()
    for name in (
        "SyntheticBinary",
        "SyntheticSelectionPackets",
        "SamaraWithinSubjectAverage",
        "SamaraTimeShift",
        "SamaraSelectionPackets",
        "BCI3PzEpochAverage",
        "BCI3SelectionPackets",
    ):
        assert name in pipelines
    models = list_models()
    assert "EEGNet" in models
    assert "BaseCNN" in models
    assert "ContextualTransformer" in models
    assert "SequenceClassifier" in models


def test_unknown_registry_entry_lists_options():
    with pytest.raises(KeyError) as pipe_exc:
        get_pipeline("DoesNotExist")
    pipe_msg = str(pipe_exc.value)
    for name in list_pipelines():
        assert name in pipe_msg

    with pytest.raises(KeyError) as model_exc:
        get_model("DoesNotExist")
    model_msg = str(model_exc.value)
    for name in list_models():
        assert name in model_msg
