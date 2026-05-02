from __future__ import annotations

from torch import Tensor
from torch.utils.data import Dataset

from tokenizer.encoder import encode_tokens
from training.mlm import create_mlm_sample


class MLMDataset(Dataset[dict[str, Tensor]]):
    tokenized_corpus: list[list[str]]
    vocab: dict[str, int]
    max_seq_len: int

    def __init__(
        self,
        tokenized_corpus: list[list[str]],
        vocab: dict[str, int],
        max_seq_len: int = 128,
    ) -> None:
        self.tokenized_corpus = tokenized_corpus
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.tokenized_corpus)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        tokens = self.tokenized_corpus[index]

        input_ids, attention_mask = encode_tokens(
            tokens,
            self.vocab,
            self.max_seq_len,
        )

        masked_ids, labels = create_mlm_sample(
            input_ids,
            self.vocab,
        )

        return {
            "input_ids": masked_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
