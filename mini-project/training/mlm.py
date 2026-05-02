import random

IGNORE_INDEX = -100

def create_mlm_sample(
    input_ids: list[int],
    vocab: dict[str, int],
    mask_prob=0.15,
) -> tuple[list[int], list[int]]:
    masked_ids, labels = list(input_ids), [IGNORE_INDEX] * len(input_ids)
    vocab_ids = list(vocab.values())
    mask_id = vocab.get("[MASK]", 0)

    for i, token_id in enumerate(input_ids):
        if random.random() >= mask_prob:
            continue
        
        labels[i] = token_id 
        rand = random.random()
        masked_ids[i] = (
            mask_id
            if rand < 0.8
            else random.choice(vocab_ids) if rand < 0.9 else token_id
        )

    return masked_ids, labels
