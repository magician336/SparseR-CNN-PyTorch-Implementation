import argparse
import os
import sys

import torch
from PIL import Image
import torchvision.transforms as T

sys.path.append(os.path.dirname(__file__))
from sparsercnn_lite import SparseRCNNLite
from utils_viz import draw_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default=None, help="Path to lite Sparse R-CNN weights .pth")
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model = SparseRCNNLite()
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    if args.weights:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state)

    img = Image.open(args.image).convert("RGB")
    x = T.ToTensor()(img)
    with torch.no_grad():
        preds = model([x.to(device)])
    preds = preds[0]
    boxes_t = preds['boxes'].detach().cpu()
    scores_t = preds['scores'].detach().cpu()
    labels_t = preds['labels'].detach().cpu()
    # filter background (last class index) for softmax case
    bg_idx = model.cfg.num_classes - 1
    keep = labels_t != bg_idx
    boxes = boxes_t[keep].numpy().tolist()
    scores = scores_t[keep].numpy().tolist()
    labels = labels_t[keep].numpy().tolist()

    vis = draw_detections(img, boxes, labels, scores, score_thr=args.score_thr, class_names=["rect", "__bg__"])
    vis.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
