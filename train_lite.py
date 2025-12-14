import argparse
import os
import sys
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
                    data_loader: DataLoader, device: torch.device, epoch: int, print_freq: int = 50):
    model.train()
    running = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = [_to_tensor_image(img).to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model._forward(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
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
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-proposals", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = SparseRCNNLite(num_classes=args.num_classes)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # DataLoaders
    train_loader = build_dataloader(args.data_root, "train", args.batch_size, args.num_workers)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

    save_checkpoint(args.output_dir, model)
    print("Training completed (lite).")


if __name__ == "__main__":
    main()
