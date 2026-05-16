import random

import torch
from torch import Tensor

import const.special_token as token
from const.mlm import IGNORE_INDEX


def create_mlm_sample(
    input_ids: Tensor,
    vocab: dict[str, int],
    mask_prob: float = 0.15,
) -> tuple[Tensor, Tensor]:
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    mask_id = vocab[token.MASK]

    vocab_ids = torch.tensor(
        tuple(vocab.values()),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    special_token_ids = torch.tensor(
        tuple(
            vocab[special]
            for special in token.SPECIAL_TOKENS
            if special in vocab
        ),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    candidate_vocab_ids = vocab_ids[~torch.isin(vocab_ids, special_token_ids)]
    special_token_mask = torch.isin(input_ids, special_token_ids)

    for i, token_id in enumerate(input_ids):
        if bool(special_token_mask[i]):
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
