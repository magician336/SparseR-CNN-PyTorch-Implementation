import argparse
import os

import cv2

from utils_viz import draw_detections

import torch
from PIL import Image
import torchvision.transforms as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="path to local config yaml; if omitted, uses model zoo config")
    parser.add_argument("--zoo-config", type=str, default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Detectron2 model zoo config key")
    parser.add_argument("--model-weights", type=str, default=None, help="Path to .pth weights")
    parser.add_argument("--score-thr", type=float, default=0.5)
    parser.add_argument("--num-classes", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.config == "config_lite.yaml":
        # Use lite inference
        import sparsercnn_lite

        model = sparsercnn_lite.SparseRCNNLite(num_classes=args.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        if args.model_weights:
            state = torch.load(args.model_weights, map_location=device)
            if 'model' in state:
                model.load_state_dict(state['model'])
            else:
                model.load_state_dict(state)
        elif os.path.exists("outputs/sparsercnn_lite_last.pth"):
            state = torch.load("outputs/sparsercnn_lite_last.pth", map_location=device)
            if 'model' in state:
                model.load_state_dict(state['model'])
            else:
                model.load_state_dict(state)

        img = Image.open(args.image).convert("RGB")
        x = T.ToTensor()(img)
        with torch.no_grad():
            preds = model([x.to(device)])
        preds = preds[0]
        boxes_t = preds['boxes'].detach().cpu()
        scores_t = preds['scores'].detach().cpu()
        labels_t = preds['labels'].detach().cpu()

        print(f"Raw predictions - boxes shape: {boxes_t.shape}, scores: {scores_t.tolist()}, labels: {labels_t.tolist()}")

        # filter background
        bg_idx = model.cfg.num_classes - 1
        keep = labels_t != bg_idx
        boxes = boxes_t[keep].numpy().tolist()
        scores = scores_t[keep].numpy().tolist()
        labels = labels_t[keep].numpy().tolist()

        print(f"After filtering background - Detected {len(boxes)} objects with scores: {scores}")

        vis = draw_detections(img, boxes, labels, scores, score_thr=args.score_thr, class_names=["rect", "__bg__"])
        vis.save(args.output)
        print(f"Saved: {args.output}")
    else:
        # Use detectron2
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        from detectron2 import model_zoo

        # Load config
        cfg = get_cfg()
        if args.config:
            cfg.merge_from_file(args.config)
        else:
            cfg.merge_from_file(model_zoo.get_config_file(args.zoo_config))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.zoo_config)
        if args.model_weights:
            cfg.MODEL.WEIGHTS = args.model_weights
        # Use generic ROI_HEADS threshold key compatible with builtin models
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thr

        predictor = DefaultPredictor(cfg)

        # Read image
        img = cv2.imread(args.image)
        outputs = predictor(img)

        # Visualize
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(args.output, out.get_image()[:, :, ::-1])
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
