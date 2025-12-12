import argparse
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import cv2

from utils_viz import draw_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="path to local config yaml; if omitted, uses model zoo config")
    parser.add_argument("--zoo-config", type=str, default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", help="Detectron2 model zoo config key")
    parser.add_argument("--model-weights", type=str, default=None, help="Path to .pth weights")
    parser.add_argument("--score-thr", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

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
