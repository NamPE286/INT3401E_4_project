from __future__ import annotations

from torch import Tensor, nn

from model.bert import BertModel
from model.config import BertConfig


class BertForMaskedLM(nn.Module):
    bert: BertModel
    mlm_head: nn.Linear

    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        self.bert = BertModel(config)
        self.mlm_head = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self.mlm_head(hidden_states)