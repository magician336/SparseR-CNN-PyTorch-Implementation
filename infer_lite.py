import argparse
import os
import sys

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import nms

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
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--skip-bg-filter", action="store_true", help="Do not filter background class labels")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="Apply NMS with this IoU threshold (<=0 to disable)")
    parser.add_argument("--max-dets", type=int, default=50, help="Maximum number of detections to keep after NMS")
    parser.add_argument("--min-size", type=float, default=4.0, help="Filter boxes smaller than this size (pixels)")
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
    total = boxes_t.shape[0]
    print(f"Total proposals: {total}")

    # Decide background index heuristically:
    # - If labels contain an index >= num_classes, treat the max label as background (softmax with bg class)
    # - Else assume focal-style (no explicit bg), unless user skips bg filtering
    bg_idx = None
    if not args.skip_bg_filter:
        max_label = int(labels_t.max().item()) if total > 0 else -1
        if max_label >= args.num_classes:
            bg_idx = max_label
            print(f"Detected background label index: {bg_idx} (>= num_classes {args.num_classes})")
            keep_bg = labels_t != bg_idx
        else:
            keep_bg = torch.ones_like(labels_t, dtype=torch.bool)
            print("No explicit background detected; not filtering by label.")
    else:
        keep_bg = torch.ones_like(labels_t, dtype=torch.bool)
        print("skip-bg-filter enabled; not filtering by background label.")

    # Apply score threshold separately for clearer debug
    keep_thr = scores_t >= args.score_thr
    print(f"Scores: min={float(scores_t.min().item()) if total>0 else 'n/a'} max={float(scores_t.max().item()) if total>0 else 'n/a'}")
    print(f"Above score_thr ({args.score_thr}): {int(keep_thr.sum().item())} / {total}")

    # Combined mask
    keep = keep_bg & keep_thr
    kept = int(keep.sum().item())
    print(f"Kept after bg+thr filtering: {kept} / {total}")
    print(f"Unique labels present: {sorted(set(labels_t.tolist())) if total>0 else []}")
    topk = min(5, total)
    if topk > 0:
        top_scores, top_idx = torch.topk(scores_t, k=topk)
        print("Top scores:", [round(float(s), 4) for s in top_scores.tolist()])
        print("Top labels:", labels_t[top_idx].tolist())

    boxes = boxes_t[keep]
    scores = scores_t[keep]
    labels = labels_t[keep]

    # Clip boxes to image size and remove tiny boxes
    w, h = img.size
    if boxes.numel() > 0:
        boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w - 1)
        boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h - 1)
        boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w - 1)
        boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h - 1)
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        sizes = torch.minimum(wh[:, 0], wh[:, 1])
        keep_size = sizes >= args.min_size
        boxes = boxes[keep_size]
        scores = scores[keep_size]
        labels = labels[keep_size]
        print(f"After size filter (min {args.min_size}px): {boxes.shape[0]} detections")

    # Apply NMS if enabled
    if args.nms_iou and args.nms_iou > 0 and boxes.numel() > 0:
        keep_nms = nms(boxes, scores, args.nms_iou)
        print(f"After NMS (IoU={args.nms_iou}): {keep_nms.numel()} detections")
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        labels = labels[keep_nms]

    # Limit number of detections
    if boxes.numel() > 0 and boxes.shape[0] > args.max_dets:
        topk = torch.topk(scores, k=args.max_dets).indices
        boxes = boxes[topk]
        scores = scores[topk]
        labels = labels[topk]
        print(f"Limited to top-{args.max_dets} detections")

    boxes = boxes.detach().cpu().numpy().tolist()
    scores = scores.detach().cpu().numpy().tolist()
    labels = labels.detach().cpu().numpy().tolist()

    vis = draw_detections(img, boxes, labels, scores, score_thr=args.score_thr, class_names=["rect", "__bg__"])
    vis.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
