from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from const.special_token import CLS, MASK, PAD, SEP
from const.mlm import IGNORE_INDEX
from training.mlm import create_mlm_sample


def test_create_mlm_sample_skips_special_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    vocab = {
        PAD: 0,
        CLS: 1,
        SEP: 2,
        MASK: 3,
        "hello": 4,
        "world": 5,
    }
    input_ids = torch.tensor(
        [vocab[CLS], vocab["hello"], vocab[SEP], vocab["world"], vocab[PAD]],
        dtype=torch.long,
    )

    monkeypatch.setattr("training.mlm.random.random", lambda: 0.0)
    monkeypatch.setattr("training.mlm.random.choice", lambda seq: seq[0])

    masked_ids, labels = create_mlm_sample(input_ids, vocab, mask_prob=1.0)

    assert isinstance(masked_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert torch.equal(
        masked_ids,
        torch.tensor(
            [vocab[CLS], vocab[MASK], vocab[SEP], vocab[MASK], vocab[PAD]],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        labels,
        torch.tensor(
            [IGNORE_INDEX, vocab["hello"], IGNORE_INDEX, vocab["world"], IGNORE_INDEX],
            dtype=torch.long,
        ),
    )
