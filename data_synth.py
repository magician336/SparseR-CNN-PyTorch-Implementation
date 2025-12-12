import argparse
import os
import random
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def gen_rect_image(w: int = 640, h: int = 480, max_rects: int = 1) -> Tuple[Image.Image, Dict[str, Any]]:
    """Generate an image with up to `max_rects` random rectangles and return image + target.

    Target format follows torchvision detection conventions:
      - boxes: FloatTensor[N, 4] in [xmin, ymin, xmax, ymax]
      - labels: Int64Tensor[N]
    """
    bg_color = (240, 240, 240)
    img = Image.new("RGB", (w, h), color=bg_color)
    draw = ImageDraw.Draw(img)

    boxes_list = []
    labels_list = []
    num_rects = max(1, max_rects)
    for _ in range(num_rects):
        rect_w = random.randint(max(20, w // 10), max(21, w // 3))
        rect_h = random.randint(max(20, h // 10), max(21, h // 3))
        x1 = random.randint(0, w - rect_w)
        y1 = random.randint(0, h - rect_h)
        x2 = x1 + rect_w
        y2 = y1 + rect_h

        color = (random.randint(60, 200), random.randint(60, 200), random.randint(60, 200))
        draw.rectangle([(x1, y1), (x2, y2)], fill=color, outline=(0, 0, 0))

        boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
        labels_list.append(0)  # single class id = 0

    boxes = torch.tensor(boxes_list, dtype=torch.float32)
    labels = torch.tensor(labels_list, dtype=torch.int64)
    target = {"boxes": boxes, "labels": labels}
    return img, target


def make_dataset(root: str, split: str, num_images: int, img_size: Tuple[int, int] = (640, 480), max_rects: int = 1) -> None:
    img_dir = os.path.join(root, split)
    _ensure_dir(img_dir)
    for i in range(num_images):
        w, h = img_size
        img, target = gen_rect_image(w=w, h=h, max_rects=max_rects)
        img.save(os.path.join(img_dir, f"img_{i:04d}.jpg"))
        # Save target as simple npz for convenience
        np.savez(os.path.join(img_dir, f"img_{i:04d}.npz"),
                 boxes=target["boxes"].numpy(), labels=target["labels"].numpy())


class SynthRectDataset(Dataset):
    def __init__(self, root: str, split: str = "train", transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.dir = os.path.join(root, split)
        self.ids = sorted([p for p in os.listdir(self.dir) if p.endswith(".jpg")])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        name = self.ids[idx]
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(self.dir, f"{stem}.jpg")
        ann_path = os.path.join(self.dir, f"{stem}.npz")

        img = Image.open(img_path).convert("RGB")
        data = np.load(ann_path)
        boxes = torch.from_numpy(data["boxes"]).float()
        labels = torch.from_numpy(data["labels"]).long()
        # normalize labels to start at 0
        labels = torch.clamp(labels, min=0)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def to_coco_format(self):
        """Convert dataset to COCO format for Detectron2."""
        coco_data = []
        for idx in range(len(self.ids)):
            name = self.ids[idx]
            stem = os.path.splitext(name)[0]
            img_path = os.path.join(self.dir, f"{stem}.jpg")
            ann_path = os.path.join(self.dir, f"{stem}.npz")

            img = Image.open(img_path).convert("RGB")
            data = np.load(ann_path)
            boxes = data["boxes"]
            labels = data["labels"]

            annotations = []
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                annotations.append({
                    "bbox": [x1, y1, w, h],  # COCO format: [x, y, w, h]
                    # ensure category ids start at 0
                    "category_id": int(max(0, int(label))),
                    "bbox_mode": 1,  # XYWH_ABS
                })

            coco_data.append({
                "file_name": img_path,
                "image_id": idx,
                "height": img.height,
                "width": img.width,
                "annotations": annotations,
            })
        return coco_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-data", action="store_true", help="Generate synthetic dataset")
    parser.add_argument("--root", default="personal/data", type=str)
    parser.add_argument("--out", dest="root", type=str, help="Alias for --root output directory")
    parser.add_argument("--num-train", default=50, type=int)
    parser.add_argument("--num-val", default=10, type=int)
    parser.add_argument("--img-size", type=int, nargs=1, help="Square image size (e.g., 800)")
    parser.add_argument("--max-rects", type=int, default=1, help="Max rectangles per image")
    args = parser.parse_args()

    if args.make_data:
        _ensure_dir(args.root)
        if args.img_size:
            img_size = (args.img_size[0], args.img_size[0])
        else:
            img_size = (640, 480)
        make_dataset(args.root, "train", args.num_train, img_size=img_size, max_rects=args.max_rects)
        make_dataset(args.root, "val", args.num_val, img_size=img_size, max_rects=args.max_rects)
        print(f"Synthetic dataset generated at {args.root}")
    else:
        print("Nothing to do. Use --make-data to generate data.")


if __name__ == "__main__":
    main()
