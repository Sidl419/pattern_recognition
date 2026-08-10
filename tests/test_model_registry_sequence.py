import pattern_recognition.models  # noqa: F401
from pattern_recognition.models import get_model, list_models


def test_sequence_models_registered():
    assert "ContextualTransformer" in list_models()
    assert "SequenceClassifier" in list_models()
    ct = get_model("ContextualTransformer")(
        eegnet={"input_feat_dim": 64, "in_channels": 1},
        d_model=32,
        nhead=4,
        num_stimulus_codes=13,
        max_flashes=32,
    )
    sc = get_model("SequenceClassifier")(
        eegnet={"input_feat_dim": 64, "in_channels": 1},
        d_model=32,
        nhead=4,
        num_stimulus_codes=17,
        max_flashes=32,
        head_mode="cell",
        n_cells=16,
    )
    assert ct is not None and sc is not None
