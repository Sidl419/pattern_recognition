import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def scaled_tanh(z):
    return 1.7159 * torch.tanh((2.0 / 3.0) * z)


def get_autoregressive_mask(size):
    """
    Returns attention mask of given size for autoregressive model.
    """
    dtype = getattr(torch, "bool", None) or torch.uint8
    res = torch.zeros(size, size, dtype=dtype)
    for i in range(size - 1):
        res[i, i + 1 :] = 1
    return res


class FlexCNN(nn.Module):
    def __init__(self, input_feat_dim, n_channels=64, num_layers=2, num_classes=2):
        super(FlexCNN, self).__init__()

        self.num_layers = num_layers
        self.num_classes = num_classes

        input_channels = n_channels
        output_channels = input_channels * 2
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(True)
        input_channels = output_channels
        output_feat_dim = input_feat_dim

        layers = []

        for _ in range(num_layers):
            output_channels = input_channels * 2
            layers.append(
                nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool1d(2, stride=2))
            input_channels = output_channels
            output_feat_dim = output_feat_dim // 2

        self.body = nn.Sequential(*layers)
        self.hook = nn.Identity()

        self.classifier_input_size = output_channels * output_feat_dim
        self.classifier = nn.Linear(self.classifier_input_size, num_classes, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.conv(x))

        x = self.body(x)

        x = self.hook(torch.flatten(x, 1))
        x = self.sig(self.classifier(x))

        return x


class CecottiCNN(nn.Module):
    def __init__(
        self, input_feat_dim, n_channels=64, num_layers=2, num_classes=2, num_fiters=10
    ):
        super(CecottiCNN, self).__init__()

        self.num_layers = num_layers
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(n_channels, num_fiters, kernel_size=1)
        self.conv2 = nn.Conv1d(
            num_fiters, 5 * num_fiters, kernel_size=13, padding="same"
        )

        self.classifier_input_size = input_feat_dim * 5 * num_fiters
        self.hook = nn.Linear(self.classifier_input_size, 100, bias=True)
        self.classifier = nn.Linear(100, num_classes, bias=True)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = scaled_tanh(self.conv1(x))
        x = scaled_tanh(self.conv2(x))

        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.sig(self.classifier(x))

        return x


class BaseCNN(nn.Module):
    def __init__(
        self,
        input_feat_dim,
        n_channels=64,
        time_kernel=13,
        num_classes=2,
        channel_filters=1,
    ):
        super(BaseCNN, self).__init__()

        self.num_classes = num_classes

        self.linear_channel = nn.Conv1d(
            n_channels, channel_filters, kernel_size=1, bias=True
        )
        self.conv = nn.Conv1d(channel_filters, 1, kernel_size=time_kernel, padding=6)
        self.bn1 = nn.BatchNorm1d(1)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)

        self.hook = nn.ReLU(True)
        self.linear_output = nn.Linear(input_feat_dim, num_classes, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_channel(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.sig(self.linear_output(x))

        return x


class PositionalEncoder(nn.Module):
    def __init__(self, dim, max_len=100, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # self.pe = torch.zeros(1, max_len, dim)
        # arg = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) /\
        #     torch.pow(scale, torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        self.pe = torch.zeros(1, dim, max_len)

        self.pe[0, 0::2, :] = torch.sin(position * div_term).transpose(0, 1)
        self.pe[0, 1::2, :] = torch.cos(position * div_term).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[:, :, : x.size(2)].to(x.device)

        return self.dropout(x)


class BaseCNNAttn(nn.Module):
    def __init__(
        self, input_feat_dim, n_channels=64, num_classes=2, num_filters=1, dropout=0.5
    ):
        super(BaseCNNAttn, self).__init__()

        self.num_classes = num_classes

        self.mask = get_autoregressive_mask(input_feat_dim)

        self.linear_channel = nn.Conv1d(
            n_channels, num_filters, kernel_size=1, bias=True
        )

        self.pos_enc = PositionalEncoder(num_filters, input_feat_dim, dropout=dropout)

        self.queries = nn.Linear(input_feat_dim, input_feat_dim)
        self.keys = nn.Linear(input_feat_dim, input_feat_dim)
        self.values = nn.Linear(input_feat_dim, input_feat_dim)

        self.attn = nn.MultiheadAttention(
            num_filters, 1, dropout=dropout, batch_first=True
        )
        self.hook = nn.ReLU(True)
        self.linear_output = nn.Linear(
            input_feat_dim * num_filters, num_classes, bias=True
        )
        self.sig = nn.Softmax()

    def forward(self, x):
        x = self.linear_channel(x)
        x = self.pos_enc(x)

        queries = self.queries(x).transpose(1, 2)
        keys = self.keys(x).transpose(1, 2)
        values = self.values(x).transpose(1, 2)

        x, self.attn_weights = self.attn(
            queries, keys, values, attn_mask=self.mask.to(x.device)
        )

        x = torch.flatten(x, 1)
        x = self.hook(x)
        x = self.sig(self.linear_output(x))

        return x


class EEGNet(nn.Module):
    def __init__(
        self,
        input_feat_dim=128,
        in_channels=64,
        num_classes=2,
        F1=8,
        D=2,
        F2=16,
        dropout=0.5,
    ):
        super(EEGNet, self).__init__()

        # First temporal convolution
        self.temporal_conv = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batch_norm1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution
        self.depthwise_conv = nn.Conv2d(
            F1, F1 * D, (in_channels, 1), groups=F1, bias=False
        )
        self.batch_norm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Separable convolution
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.batch_norm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # Classification layer
        self.classifier = nn.Linear(F2 * (input_feat_dim // 32), num_classes)

    def extract_features(self, x):
        """Return the single-flash feature vector before the classifier.

        This keeps ``forward`` backwards compatible while allowing sequence
        models to reuse EEGNet as a flash-epoch encoder.
        """
        # x shape: (batch, channels, samples)
        x = x.unsqueeze(1)  # (batch, 1, channels, samples)

        x = self.temporal_conv(x)
        x = self.batch_norm1(x)

        x = self.depthwise_conv(x)
        x = self.batch_norm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.separable_conv(x)
        x = self.batch_norm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        return self.classifier(self.extract_features(x))


class _NonNativeMultiheadAttention(nn.MultiheadAttention):
    """Multi-head attention that skips Torch 2.0's native fused kernel.

    ``torch._native_multi_head_attention`` converts bool padding masks to the
    query dtype then warns when converting back inside SDPA. Using distinct
    query/key/value tensor identities forces the portable
    ``F.multi_head_attention_forward`` path (same math, no warning).
    """

    def forward(self, query, key, value, *args, **kwargs):  # type: ignore[override]
        if query is key and key is value:
            key = value = query.clone()
        return super().forward(query, key, value, *args, **kwargs)


class _NonFusedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Encoder layer that never enters ``_transformer_encoder_layer_fwd``.

    The fused eval path has the same padding-mask dtype warning as native MHA
    on Torch 2.0. The Python ``_sa_block`` / ``_ff_block`` path is equivalent.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        del is_causal  # causal masking is supplied via ``src_mask`` by callers
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class P300SequenceEncoder(nn.Module):
    """Embed every P300 flash with EEGNet and contextualise a character sequence.

    Parameters
    ----------
    eegnet:
        Existing single-trial :class:`EEGNet`. Its convolutional feature
        extractor is reused; its binary classifier is not used here.
    d_model:
        Transformer embedding width.

    Notes
    -----
    ``epochs`` has shape ``[batch, flashes, channels, samples]``; on BCI III
    this is normally ``[B, 180, 64, 72]``. ``stimulus_codes`` contains the
    actual row/column IDs 1..12 (0 is reserved for padding). ``repetitions``
    contains 0..14. The model never invents a speller schedule.
    """

    def __init__(
        self,
        eegnet: EEGNet,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.15,
        max_flashes: int = 180,
        max_repetitions: int = 15,
        num_stimulus_codes: int = 13,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.eegnet = eegnet
        self.max_flashes = max_flashes
        self.max_repetitions = max_repetitions
        self.num_stimulus_codes = num_stimulus_codes
        feature_dim = eegnet.classifier.in_features
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.code_embedding = nn.Embedding(num_stimulus_codes, d_model, padding_idx=0)
        self.repetition_embedding = nn.Embedding(max_repetitions, d_model)
        self.position_embedding = nn.Embedding(max_flashes, d_model)

        layer = _NonFusedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        safe_attn = _NonNativeMultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        safe_attn.load_state_dict(layer.self_attn.state_dict())
        layer.self_attn = safe_attn
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(
        self,
        epochs: torch.Tensor,
        stimulus_codes: torch.Tensor,
        repetitions: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if epochs.ndim != 4:
            raise ValueError(
                "epochs must have shape [batch, flashes, channels, samples]"
            )
        batch, flashes, channels, samples = epochs.shape
        if flashes > self.max_flashes:
            raise ValueError(
                f"got {flashes} flashes; max_flashes is {self.max_flashes}"
            )
        if stimulus_codes.shape != (batch, flashes) or repetitions.shape != (
            batch,
            flashes,
        ):
            raise ValueError(
                "stimulus_codes and repetitions must have shape [batch, flashes]"
            )
        max_code = int(stimulus_codes.max().item())
        if stimulus_codes.min() < 0 or max_code >= self.num_stimulus_codes:
            raise ValueError(
                f"stimulus_codes must be in 0..{self.num_stimulus_codes - 1}; "
                "zero is padding only"
            )
        if repetitions.min() < 0 or repetitions.max() >= self.max_repetitions:
            raise ValueError("repetitions outside configured range")

        if valid_mask is None:
            valid_mask = torch.ones(
                batch, flashes, dtype=torch.bool, device=epochs.device
            )

        flat_epochs = epochs.reshape(batch * flashes, channels, samples)
        flash_features = self.eegnet.extract_features(flat_epochs)
        x = self.feature_projection(flash_features).reshape(batch, flashes, -1)

        position = torch.arange(flashes, device=epochs.device).unsqueeze(0)
        x = x + self.code_embedding(stimulus_codes.long())
        x = x + self.repetition_embedding(repetitions.long())
        x = x + self.position_embedding(position)

        attention_mask = None
        if causal:
            attention_mask = torch.triu(
                torch.ones(flashes, flashes, dtype=torch.bool, device=epochs.device),
                diagonal=1,
            )

        x = self.transformer(
            x,
            mask=attention_mask,
            src_key_padding_mask=~valid_mask.to(dtype=torch.bool),
        )
        return x, valid_mask.to(dtype=torch.bool)


class ContextualTransformer(nn.Module):
    """Flash-wise contextual P300 detector: P(target_i | sequence)."""

    def __init__(self, sequence_encoder: P300SequenceEncoder):
        super().__init__()
        self.sequence_encoder = sequence_encoder
        d_model = sequence_encoder.code_embedding.embedding_dim
        self.flash_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``flash_logits [B,S]`` and ``valid_mask [B,S]``."""
        embeddings, valid_mask = self.sequence_encoder(*args, **kwargs)
        return self.flash_classifier(embeddings).squeeze(-1), valid_mask

    @staticmethod
    def loss(
        flash_logits: torch.Tensor,
        flash_targets: torch.Tensor,
        valid_mask: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked weighted binary loss; estimate ``pos_weight`` on train only."""
        return F.binary_cross_entropy_with_logits(
            flash_logits[valid_mask],
            flash_targets.float()[valid_mask],
            pos_weight=pos_weight,
        )


class SequenceClassifier(nn.Module):
    """Direct character decoder: P(row, column | complete flash sequence)."""

    def __init__(
        self,
        sequence_encoder: P300SequenceEncoder,
        *,
        head_mode: str = "rowcol",
        include_character_head: bool = True,
        n_cells: int = 16,
    ):
        super().__init__()
        self.sequence_encoder = sequence_encoder
        self.head_mode = head_mode
        self.n_cells = n_cells
        d_model = sequence_encoder.code_embedding.embedding_dim
        self.pool_attention = nn.Linear(d_model, 1)

        if head_mode == "rowcol":
            self.row_classifier = nn.Linear(d_model, 6)
            self.column_classifier = nn.Linear(d_model, 6)
            self.character_classifier = (
                nn.Linear(d_model, 36) if include_character_head else None
            )
        elif head_mode == "cell":
            self.row_classifier = None
            self.column_classifier = None
            self.character_classifier = nn.Linear(d_model, n_cells)
        else:
            raise ValueError(f"head_mode must be 'rowcol' or 'cell', got {head_mode!r}")

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        embeddings, valid_mask = self.sequence_encoder(*args, **kwargs)
        weights = self.pool_attention(embeddings).squeeze(-1)
        weights = weights.masked_fill(~valid_mask, float("-inf")).softmax(dim=1)
        pooled = torch.einsum("bs,bsd->bd", weights, embeddings)

        output: dict[str, torch.Tensor] = {}
        if self.row_classifier is not None:
            output["row_logits"] = self.row_classifier(pooled)
        if self.column_classifier is not None:
            output["column_logits"] = self.column_classifier(pooled)
        if self.character_classifier is not None:
            output["character_logits"] = self.character_classifier(pooled)
        return output

    @staticmethod
    def loss(
        output: dict[str, torch.Tensor],
        row_targets: Optional[torch.Tensor] = None,
        column_targets: Optional[torch.Tensor] = None,
        character_targets: Optional[torch.Tensor] = None,
        character_weight: float = 0.25,
    ) -> torch.Tensor:
        if "row_logits" in output and "column_logits" in output:
            if row_targets is None or column_targets is None:
                raise ValueError(
                    "row_targets and column_targets required for rowcol head"
                )
            loss = F.cross_entropy(output["row_logits"], row_targets.long())
            loss = loss + F.cross_entropy(
                output["column_logits"], column_targets.long()
            )
            if character_targets is not None and "character_logits" in output:
                loss = loss + character_weight * F.cross_entropy(
                    output["character_logits"], character_targets.long()
                )
            return loss

        if "character_logits" in output:
            if character_targets is None:
                raise ValueError("character_targets required for cell head")
            return F.cross_entropy(output["character_logits"], character_targets.long())

        raise ValueError("output dict has no active classification heads")


class DeepConvNet(nn.Module):
    def __init__(self, input_feat_dim=128, in_channels=64, num_classes=2):
        super(DeepConvNet, self).__init__()

        # First conv block
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(25, 25, (in_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
        )

        # Second conv block
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
        )

        # Third conv block
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
        )

        # Fourth conv block
        self.block4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
        )

        # Flatten and classify
        # Compute flattened feature size after pooling (assuming input_feat_dim input length)
        def feature_size(input_feat_dim):
            for _ in range(4):
                input_feat_dim = (
                    input_feat_dim + 2 * 2 - 5
                ) + 1  # conv 5 kernel size, pad 2
                input_feat_dim = input_feat_dim // 2  # maxpool with kernel 2, stride 2
            return input_feat_dim

        self.feature_len = feature_size(input_feat_dim) * 200

        self.classifier = nn.Linear(self.feature_len, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, samples)
        x = x.unsqueeze(1)  # (batch, 1, channels, samples)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
