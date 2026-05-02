from __future__ import annotations

import torch
from torch import Tensor

import const.special_token as token


def encode_tokens(
    tokens: list[str],
    vocab: dict[str, int],
    max_seq_len: int = 128,
) -> tuple[Tensor, Tensor]:
    if max_seq_len < 2:
        raise ValueError("max_len must be at least 2")

    sequence_tokens: list[str] = [token.CLS] + tokens[: max_seq_len - 2] + [token.SEP]

    unk_id = vocab[token.UNK]
    pad_id = vocab[token.PAD]

    encoded_tokens = torch.tensor(
        [vocab.get(token, unk_id) for token in sequence_tokens],
        dtype=torch.long,
    )

    input_ids = torch.full((max_seq_len,), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(max_seq_len, dtype=torch.long)

    seq_len = encoded_tokens.size(0)
    input_ids[:seq_len] = encoded_tokens
    attention_mask[:seq_len] = 1

    return input_ids, attention_mask
