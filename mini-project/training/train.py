from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from const.mlm import IGNORE_INDEX
from model.config import BertConfig
from model.mlm import BertForMaskedLM
from scripts.load_corpus import load_tokenized_corpus
from tokenizer.build_vocab import build_vocab
from training.dataset import MLMDataset


EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
LOG_EVERY = 50
SAVE_PATH = PROJECT_ROOT / "checkpoints" / "bert_mlm.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(
    model: BertForMaskedLM,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    update_steps = 0

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)

        if torch.all(batch["labels"] == IGNORE_INDEX):
            continue

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        update_steps += 1

        if step % log_every == 0:
            avg_loss = total_loss / max(update_steps, 1)
            print(f"epoch {epoch} step {step}/{len(dataloader)} loss {avg_loss:.4f}")

    return total_loss / max(update_steps, 1)


def save_checkpoint(
    path: Path,
    model: BertForMaskedLM,
    vocab: dict[str, int],
    config: BertConfig,
    epoch: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": config,
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def main() -> None:
    device = get_device()

    tokenized_corpus = load_tokenized_corpus()
    vocab = build_vocab(tokenized_corpus)
    config = BertConfig(vocab_size=len(vocab))

    dataset = MLMDataset(
        tokenized_corpus=tokenized_corpus,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = BertForMaskedLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    print(
        f"training on {device} | samples {len(dataset)} | "
        f"vocab {len(vocab)} | batches {len(dataloader)}"
    )

    final_loss = 0.0
    for epoch in range(1, EPOCHS + 1):
        final_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            log_every=LOG_EVERY,
        )
        print(f"epoch {epoch} complete loss {final_loss:.4f}")

    save_checkpoint(
        path=SAVE_PATH,
        model=model,
        vocab=vocab,
        config=config,
        epoch=EPOCHS,
        loss=final_loss,
    )
    print(f"saved checkpoint to {SAVE_PATH}")


if __name__ == "__main__":
    main()
