from pattern_recognition.models import get_model
from pattern_recognition.training.sequence_loop import selection_classifier_n_classes


def test_selection_classifier_n_classes_rowcol():
    model = get_model("SequenceClassifier")(
        eegnet={"input_feat_dim": 64, "in_channels": 1},
        d_model=32,
        nhead=4,
        num_layers=1,
        num_stimulus_codes=13,
        max_flashes=24,
        max_repetitions=15,
        head_mode="rowcol",
    )
    assert selection_classifier_n_classes(model) == 36


def test_selection_classifier_n_classes_cell_uses_n_cells():
    model = get_model("SequenceClassifier")(
        eegnet={"input_feat_dim": 64, "in_channels": 1},
        d_model=32,
        nhead=4,
        num_layers=1,
        num_stimulus_codes=17,
        max_flashes=32,
        max_repetitions=15,
        head_mode="cell",
        n_cells=16,
    )
    assert selection_classifier_n_classes(model) == 16

    model4 = get_model("SequenceClassifier")(
        eegnet={"input_feat_dim": 64, "in_channels": 1},
        d_model=32,
        nhead=4,
        num_layers=1,
        num_stimulus_codes=5,
        max_flashes=8,
        max_repetitions=15,
        head_mode="cell",
        n_cells=4,
    )
    assert selection_classifier_n_classes(model4) == 4
