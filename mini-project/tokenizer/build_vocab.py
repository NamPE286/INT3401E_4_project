from collections import Counter
from const.special_token import SPECIAL_TOKENS


def build_vocab(
    tokenized_corpus: list[list[str]], vocab_size: int = 10000
) -> dict[str, int]:
    counter = Counter()

    for tokens in tokenized_corpus:
        counter.update(tokens)

    vocab: dict[str, int] = {}

    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    most_common = counter.most_common(vocab_size - len(vocab))

    for token, _ in most_common:
        vocab[token] = len(vocab)

    return vocab
