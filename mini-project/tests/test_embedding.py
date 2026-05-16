from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.config import BertConfig
from model.embedding import BertEmbeddings


def test_bert_embeddings_returns_expected_shape() -> None:
    config = BertConfig(
        vocab_size=12,
        max_seq_len=8,
        hidden_size=6,
        dropout=0.0,
    )
    embeddings = BertEmbeddings(config)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ],
        dtype=torch.long,
    )

    output = embeddings(input_ids)

    assert output.shape == (2, 4, config.hidden_size)


def test_bert_embeddings_adds_position_embeddings_before_layer_norm() -> None:
    config = BertConfig(
        vocab_size=4,
        max_seq_len=3,
        hidden_size=3,
        dropout=0.0,
    )
    embeddings = BertEmbeddings(config)
    input_ids = torch.tensor([[1, 1, 1]], dtype=torch.long)

    with torch.no_grad():
        embeddings.token_embedding.weight.zero_()
        embeddings.position_embedding.weight.copy_(
            torch.tensor(
                [
                    [1.0, 3.0, 5.0],
                    [2.0, 6.0, 7.0],
                    [5.0, 8.0, 9.0],
                ]
            )
        )
        embeddings.layer_norm.weight.fill_(1.0)
        embeddings.layer_norm.bias.zero_()

    output = embeddings(input_ids)
    expected = embeddings.layer_norm(
        embeddings.position_embedding(torch.tensor([[0, 1, 2]], dtype=torch.long))
    )

    assert torch.allclose(output, expected)
    assert not torch.allclose(output[:, 0], output[:, 1])
