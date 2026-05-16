from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.bert import BertModel
from model.config import BertConfig
from model.mlm import BertForMaskedLM


def test_bert_model_forward_shape() -> None:
    config = BertConfig(
        vocab_size=100,
        max_seq_len=16,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        intermediate_size=64,
        dropout=0.1,
    )

    model = BertModel(config)

    input_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 5, 0, 0],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert output.shape == (2, 4, 32)


def test_bert_model_supports_full_max_seq_len() -> None:
    config = BertConfig(
        vocab_size=100,
        max_seq_len=8,
        hidden_size=32,
        num_heads=4,
        num_layers=1,
        intermediate_size=64,
        dropout=0.1,
    )

    model = BertModel(config)

    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 0]],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 0]],
        dtype=torch.long,
    )

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert output.shape == (1, 8, 32)
    
def test_bert_for_masked_lm_forward_shape() -> None:
    config = BertConfig(
        vocab_size=100,
        max_seq_len=16,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        intermediate_size=64,
        dropout=0.1,
    )

    model = BertForMaskedLM(config)

    input_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 5, 0, 0],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert logits.shape == (2, 4, 100)