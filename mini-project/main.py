from pathlib import Path

import torch

import const.special_token as token
from model.mlm import BertForMaskedLM
from tokenizer.basic_tokenizer import tokenize
from tokenizer.encoder import encode_tokens
from training.train import get_device


ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "bert_mlm.pt"


def main() -> None:
    device = get_device()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    vocab, config = ckpt["vocab"], ckpt["config"]
    id_to_token = {idx: word for word, idx in vocab.items()}

    model = BertForMaskedLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    text = input("text (_ = mask): ")
    words = [token.MASK if word == "_" else word for word in tokenize(text)]
    input_ids, attention_mask = encode_tokens(words, vocab, config.max_seq_len)

    with torch.no_grad():
        logits = model(input_ids[None].to(device), attention_mask[None].to(device))[0]

    for i, word in enumerate(words, start=1):
        if word == token.MASK:
            words[i - 1] = id_to_token[int(logits[i].argmax())]

    print(" ".join(words))


if __name__ == "__main__":
    main()
