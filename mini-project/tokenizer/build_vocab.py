from collections import Counter
from typing import Dict, List

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]


def build_vocab(
    tokenized_corpus: List[List[str]], vocab_size: int = 10000
) -> Dict[str, int]:
    counter = Counter()

    for tokens in tokenized_corpus:
        counter.update(tokens)

    vocab: Dict[str, int] = {}

    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    most_common = counter.most_common(vocab_size - len(vocab))

    for token, _ in most_common:
        vocab[token] = len(vocab)

    return vocab
