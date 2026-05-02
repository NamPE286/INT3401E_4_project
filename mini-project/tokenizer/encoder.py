import const.special_token as token


def encode_tokens(
    tokens: list[str],
    vocab: dict[str, int],
    max_seq_len: int = 128,
) -> tuple[list[int], list[int]]:
    if max_seq_len < 2:
        raise ValueError("max_len must be at least 2")

    sequence_tokens: list[str] = [token.CLS] + tokens[: max_seq_len - 2] + [token.SEP]

    unk_id = vocab[token.UNK]
    pad_id = vocab[token.PAD]

    input_ids = [vocab.get(token, unk_id) for token in sequence_tokens]

    attention_mask = [1] * len(input_ids)

    pad_count = max_seq_len - len(input_ids)

    input_ids.extend([pad_id] * pad_count)
    attention_mask.extend([0] * pad_count)

    return input_ids, attention_mask
