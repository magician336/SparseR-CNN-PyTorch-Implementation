import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from data_synth import SynthRectDataset


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def build_model(use_pretrained: bool, num_classes: int = 2):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    if use_pretrained:
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
        except Exception:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq: int = 20):
    model.train()
    import time
    import math
    from collections import deque

    losses_deque = deque(maxlen=print_freq)
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        losses_deque.append(losses.item())
        if (i + 1) % print_freq == 0:
            eta = (time.time() - start) / (i + 1) * (len(data_loader) - i - 1)
            print(f"Epoch {epoch} [{i+1}/{len(data_loader)}] loss={sum(losses_deque)/len(losses_deque):.4f} ETA={eta:.1f}s")


@torch.no_grad()
def evaluate_aprox(model, data_loader, device):
    model.eval()
    cnt = 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        _ = model(images)
        cnt += 1
        if cnt >= 3:
            break
    print("Eval (approx): ran a few batches to validate the pipeline.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="personal/data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="personal/outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Dataset & DataLoader
    transform = T.ToTensor()
    train_set = SynthRectDataset(args.data_root, split="train", transforms=transform)
    val_set = SynthRectDataset(args.data_root, split="val", transforms=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Model
    model = build_model(use_pretrained=args.use_pretrained, num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        evaluate_aprox(model, val_loader, device)

    # Save final weights
    out_path = os.path.join(args.output_dir, "model_last.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights to {out_path}")


if __name__ == "__main__":
    main()
