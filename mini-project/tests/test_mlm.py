from pathlib import Path
import sys

import pytest

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
    input_ids = [vocab[CLS], vocab["hello"], vocab[SEP], vocab["world"], vocab[PAD]]

    monkeypatch.setattr("training.mlm.random.random", lambda: 0.0)
    monkeypatch.setattr("training.mlm.random.choice", lambda seq: seq[0])

    masked_ids, labels = create_mlm_sample(input_ids, vocab, mask_prob=1.0)

    assert masked_ids[0] == vocab[CLS]
    assert masked_ids[2] == vocab[SEP]
    assert masked_ids[4] == vocab[PAD]
    assert labels[0] == IGNORE_INDEX
    assert labels[2] == IGNORE_INDEX
    assert labels[4] == IGNORE_INDEX
    assert labels[1] == vocab["hello"]
    assert labels[3] == vocab["world"]
