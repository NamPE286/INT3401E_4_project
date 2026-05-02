from dataclasses import dataclass


@dataclass(frozen=True)
class MiniBertConfig:
    vocab_size: int
    max_seq_len: int = 128
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    intermediate_size: int = 512
    dropout: float = 0.1