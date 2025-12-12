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
    parser.add_argument("--num-classes", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model = SparseRCNNLite(num_classes=args.num_classes)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    if args.weights:
        try:
            state = torch.load(args.weights, map_location=device)
            if 'model' in state:
                missing_keys, unexpected_keys = model.load_state_dict(state['model'], strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys in state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                print(f"Loaded Detectron2 checkpoint from {args.weights}")
            else:
                missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys in state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                print(f"Loaded direct state_dict from {args.weights}")
        except Exception as e:
            print(f"Error loading weights from {args.weights}: {e}")
            print("Proceeding with randomly initialized weights.")
    else:
        print("No weights provided, using randomly initialized model.")

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
