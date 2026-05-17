"""
Fine-tune Mental-RoBERTa (sequence classification) on sorted_multimodal text folders.

Uses a plain PyTorch loop (no Hugging Face Trainer) so TensorFlow/Keras is never
loaded — avoids Keras 3 / tf_keras errors when TF happens to be installed.

Layout:
  sorted_multimodal/text/training/PTSD/*.txt
  sorted_multimodal/text/training/NO PTSD/*.txt
  sorted_multimodal/text/validation/...

Recommended:

  python train_text_explainer.py --data-root sorted_multimodal/text --epochs 12 --batch-size 8 --early-stopping-patience 2

Larger effective batch:

  python train_text_explainer.py --data-root sorted_multimodal/text --epochs 12 --batch-size 4 --gradient-accumulation-steps 2 --early-stopping-patience 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


LABELS = ["NO PTSD", "PTSD"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}


def collect_samples(split_root: Path) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    if not split_root.is_dir():
        return texts, labels
    for label_dir in sorted(split_root.iterdir()):
        if not label_dir.is_dir():
            continue
        folder_name = label_dir.name.strip()
        if folder_name not in LABEL2ID:
            continue
        lid = LABEL2ID[folder_name]
        for fp in sorted(label_dir.glob("*.txt")):
            try:
                raw = fp.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if len(raw) < 20:
                continue
            texts.append(raw)
            labels.append(lid)
    return texts, labels


class EncodedDataset(Dataset):
    def __init__(self, encodings: dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def collate_encoded(batch: List[dict]) -> dict:
    keys_feat = [k for k in batch[0].keys() if k != "labels"]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_feat}
    out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


@torch.no_grad()
def eval_loss_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(**batch, labels=labels)
        loss = float(out.loss.item())
        bs = labels.size(0)
        total += loss * bs
        n += bs
    model.train()
    return total / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PTSD text explainer (Mental RoBERTa)")
    ap.add_argument(
        "--data-root",
        type=str,
        default="sorted_multimodal/text",
        help="Root containing training/ and validation/ splits",
    )
    ap.add_argument("--output", type=str, default="saved_models/text_ptsd_explainer")
    ap.add_argument("--base-model", type=str, default="mental/mental-roberta-base")
    ap.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Max epochs (early stopping may finish sooner).",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop if eval loss does not improve for N epochs (0 = disabled). Needs validation data.",
    )
    ap.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable mixed precision on CUDA.",
    )
    args = ap.parse_args()

    root = Path(args.data_root)
    train_tx, train_y = collect_samples(root / "training")
    val_tx, val_y = collect_samples(root / "validation")

    if len(train_tx) < 10:
        raise SystemExit(f"Too few training samples ({len(train_tx)}). Check --data-root ({root}).")

    # Torch-only HF imports (avoid transformers.trainer → TensorFlow stack).
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    enc_train = tokenizer(
        train_tx,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors=None,
    )
    ds_train = EncodedDataset(enc_train, train_y)

    ds_eval = None
    eval_loader = None
    if val_tx:
        enc_val = tokenizer(
            val_tx,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors=None,
        )
        ds_eval = EncodedDataset(enc_val, val_y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(not args.no_fp16 and device.type == "cuda")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label={i: n for i, n in enumerate(LABELS)},
        label2id=LABEL2ID,
    ).to(device)

    grad_accum = max(1, args.gradient_accumulation_steps)

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_encoded,
        drop_last=False,
    )
    if ds_eval is not None:
        eval_loader = DataLoader(
            ds_eval,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_encoded,
            drop_last=False,
        )

    steps_per_epoch = max(
        1, (len(train_loader) + grad_accum - 1) // grad_accum
    )
    total_optimizer_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.1 * total_optimizer_steps))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    from transformers.optimization import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_optimizer_steps, 1),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_eval = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    stall = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(**batch, labels=labels)
                loss = out.loss / grad_accum

            scaler.scale(loss).backward()
            epoch_loss += float(out.loss.detach().item()) * labels.size(0)
            epoch_n += labels.size(0)

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if len(train_loader) % grad_accum != 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        train_avg = epoch_loss / max(epoch_n, 1)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_avg:.4f}")

        if eval_loader is not None:
            vloss = eval_loss_epoch(model, eval_loader, device, use_amp)
            print(f"             eval_loss={vloss:.4f}")

            if vloss < best_eval - 1e-6:
                best_eval = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1

            if args.early_stopping_patience > 0 and stall >= args.early_stopping_patience:
                print(
                    f"Early stopping (no eval improvement for {args.early_stopping_patience} epochs)."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Loaded best checkpoint (eval_loss={best_eval:.4f}).")

    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"Saved text explainer to {out_dir.resolve()}")
    if use_amp:
        print("Training used fp16 autocast on CUDA.")


if __name__ == "__main__":
    main()
