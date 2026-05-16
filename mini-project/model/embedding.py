from __future__ import annotations

import torch
from torch import Tensor, nn

from model.config import BertConfig


class BertEmbeddings(nn.Module):
    token_embedding: nn.Embedding
    position_embedding: nn.Embedding
    layer_norm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=config.max_seq_len,
            embedding_dim=config.hidden_size,
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, seq_len = input_ids.shape

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)

        token_vectors: Tensor = self.token_embedding(input_ids)
        position_vectors: Tensor = self.position_embedding(position_ids)

        embeddings = token_vectors + position_vectors
        embeddings = self.layer_norm(embeddings)

        return self.dropout(embeddings)