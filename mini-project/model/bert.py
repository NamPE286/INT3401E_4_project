from __future__ import annotations

from torch import Tensor, nn

from model.config import BertConfig
from model.embedding import BertEmbeddings


class BertModel(nn.Module):
    embeddings: BertEmbeddings
    encoder: nn.TransformerEncoder

    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        self.embeddings = BertEmbeddings(config)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        embeddings = self.embeddings(input_ids)

        key_padding_mask = attention_mask == 0

        return self.encoder(
            embeddings,
            src_key_padding_mask=key_padding_mask,
        )