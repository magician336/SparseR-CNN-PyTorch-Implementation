import argparse
import os
import sys
import csv
import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Ensure root is on path to import shared modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparsercnn_lite import SparseRCNNLite
from data_synth import SynthRectDataset


def collate_fn(batch: Tuple):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def build_dataloader(data_root: str, split: str, batch_size: int, num_workers: int = 2):
    # SynthRectDataset does not accept a transform argument; convert to tensor later
    ds = SynthRectDataset(data_root, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                        num_workers=num_workers, collate_fn=collate_fn)
    return loader


def _to_tensor_image(x):
    # Convert dataset image to torch tensor in [0,1], shape [C,H,W]
    if isinstance(x, torch.Tensor):
        return x.float()
    try:
        return T.ToTensor()(x)
    except Exception:
        import numpy as np
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
            if t.ndim == 2:
                t = t.unsqueeze(0)
            elif t.ndim == 3:
                t = t.permute(2, 0, 1)
            return t.float() / 255.0
        raise


def train_one_epoch(model: SparseRCNNLite, optimizer: torch.optim.Optimizer,
                    data_loader: DataLoader, device: torch.device, epoch: int, print_freq: int = 50,
                    csv_writer: "csv.DictWriter | None" = None, csv_flush: bool = True):
    model.train()
    running = 0.0
    running_parts = {}
    t0 = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = [_to_tensor_image(img).to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model._forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

        # Track component losses
        parts = {}
        for k, v in loss_dict.items():
            try:
                parts[k] = float(v.detach().item())
            except Exception:
                parts[k] = float(v)
            running_parts[k] = running_parts.get(k, 0.0) + parts[k]

        # CSV logging per-iteration
        if csv_writer is not None:
            lr = optimizer.param_groups[0].get("lr", None)
            row = {
                "epoch": epoch,
                "iter": i + 1,
                "iter_total": len(data_loader),
                "iter_str": f"{i+1}/{len(data_loader)}",
                "total_loss": float(loss.detach().item()),
                "avg_total_loss": running / (i + 1),
                "lr": float(lr) if lr is not None else "",
                "elapsed_sec": round(time.time() - t0, 4),
            }
            # Add parts and running averages
            for k in sorted(parts.keys()):
                row[k] = parts[k]
                row[f"avg_{k}"] = running_parts[k] / (i + 1)
            csv_writer.writerow(row)
            if csv_flush:
                try:
                    csv_writer._writerows  # type: ignore[attr-defined]
                except Exception:
                    pass

        if (i + 1) % print_freq == 0 or (i + 1) == len(data_loader):
            avg = running / (i + 1)
            print(f"Epoch {epoch} [{i+1}/{len(data_loader)}] loss: {avg:.4f}")


def save_checkpoint(output_dir: str, model: SparseRCNNLite):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_final.pth")
    torch.save({"model": model.state_dict()}, path)
    print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-proposals", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--csv-path", type=str, default=None, help="Where to write loss CSV (default: <output-dir>/loss_log.csv)")
    parser.add_argument("--csv-every", type=int, default=1, help="Write one CSV row every N iterations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = SparseRCNNLite(num_classes=args.num_classes)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # DataLoaders
    train_loader = build_dataloader(args.data_root, "train", args.batch_size, args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = args.csv_path or os.path.join(args.output_dir, "loss_log.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = None

    try:
        # Initialize CSV header after first forward so we know exact loss_dict keys
        for epoch in range(1, args.epochs + 1):
            if csv_writer is None:
                # Peek one batch to derive header keys
                model.train()
                images, targets = next(iter(train_loader))
                images = [_to_tensor_image(img).to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.no_grad():
                    loss_dict = model._forward(images, targets)
                part_keys = sorted(loss_dict.keys())
                fieldnames = [
                    "epoch",
                    "iter",
                    "iter_total",
                    "iter_str",
                    "total_loss",
                    "avg_total_loss",
                    "lr",
                    "elapsed_sec",
                ]
                for k in part_keys:
                    fieldnames.append(k)
                    fieldnames.append(f"avg_{k}")
                csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
                csv_writer.writeheader()
                csv_f.flush()

            # Run training epoch (log every iteration inside)
            train_one_epoch(
                model,
                optimizer,
                train_loader,
                device,
                epoch,
                print_freq=50,
                csv_writer=_EveryNWriter(csv_writer, every=max(1, int(args.csv_every))),
                csv_flush=True,
            )
            csv_f.flush()
    finally:
        try:
            csv_f.close()
        except Exception:
            pass

    save_checkpoint(args.output_dir, model)
    print("Training completed (lite).")


class _EveryNWriter:
    """Wrapper that only writes every N calls to writerow."""

    def __init__(self, writer: csv.DictWriter, every: int = 1):
        self.writer = writer
        self.every = max(1, int(every))
        self._count = 0

    def writerow(self, row: dict):
        self._count += 1
        if (self._count % self.every) == 0:
            self.writer.writerow(row)


if __name__ == "__main__":
    main()
