import random

import const.special_token as token
from const.mlm import IGNORE_INDEX


def create_mlm_sample(
    input_ids: list[int],
    vocab: dict[str, int],
    mask_prob: float = 0.15,
) -> tuple[list[int], list[int]]:
    masked_ids = list(input_ids)
    labels = [IGNORE_INDEX] * len(input_ids)

    mask_id = vocab[token.MASK]

    special_token_ids = {
        vocab[special]
        for special in token.SPECIAL_TOKENS
        if special in vocab
    }

    candidate_vocab_ids = [
        token_id
        for token_id in vocab.values()
        if token_id not in special_token_ids
    ]

    for i, token_id in enumerate(input_ids):
        if token_id in special_token_ids:
            continue

        if random.random() >= mask_prob:
            continue

        labels[i] = token_id

        rand = random.random()

        if rand < 0.8:
            masked_ids[i] = mask_id
        elif rand < 0.9:
            masked_ids[i] = random.choice(candidate_vocab_ids)
        else:
            masked_ids[i] = token_id

    return masked_ids, labels