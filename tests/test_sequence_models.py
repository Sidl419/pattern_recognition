import torch
from pattern_recognition.models.cnn import (
    EEGNet,
    P300SequenceEncoder,
    SequenceClassifier,
)


def test_eegnet_extract_features_then_forward():
    m = EEGNet(input_feat_dim=64, in_channels=1, num_classes=2)
    x = torch.randn(4, 1, 64)
    feats = m.extract_features(x)
    out = m(x)
    assert feats.ndim == 2 and out.shape == (4, 2)


def test_sequence_encoder_accepts_samara_codes():
    eeg = EEGNet(input_feat_dim=64, in_channels=1, num_classes=2)
    enc = P300SequenceEncoder(
        eeg, d_model=32, nhead=4, num_stimulus_codes=17, max_flashes=32
    )
    B, S, C, T = 2, 16, 1, 64
    epochs = torch.randn(B, S, C, T)
    codes = torch.arange(1, 17).repeat(B, 1)[:, :S]
    reps = torch.zeros(B, S, dtype=torch.long)
    emb, mask = enc(epochs, codes, reps)
    assert emb.shape == (B, S, 32) and mask.shape == (B, S)


def test_sequence_classifier_cell_head_samara():
    eeg = EEGNet(input_feat_dim=64, in_channels=1, num_classes=2)
    enc = P300SequenceEncoder(
        eeg, d_model=32, nhead=4, num_stimulus_codes=17, max_flashes=32
    )
    clf = SequenceClassifier(enc, head_mode="cell", n_cells=16)
    B, S = 2, 16
    epochs = torch.randn(B, S, 1, 64)
    codes = torch.arange(1, 17).unsqueeze(0).expand(B, -1)
    reps = torch.zeros(B, S, dtype=torch.long)
    out = clf(epochs, codes, reps)
    assert out["character_logits"].shape == (B, 16)
    assert "row_logits" not in out
