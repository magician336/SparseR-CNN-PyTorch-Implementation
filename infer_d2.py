import argparse
import os
import sys

import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Ensure projects/SparseRCNN is importable and registers meta-arch
PROJ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "projects", "SparseRCNN")
if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)
# This import registers SparseRCNN and adds its config keys
from sparsercnn import add_sparsercnn_config  # noqa: F401


def build_cfg(config_file: str, num_classes: int = None, num_proposals: int = None,
              ims_per_batch_test: int = None, score_thr: float = 0.5):
    cfg = get_cfg()
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(config_file)
    if num_classes is not None:
        cfg.MODEL.SparseRCNN.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    if num_proposals is not None:
        cfg.MODEL.SparseRCNN.NUM_PROPOSALS = num_proposals
    if ims_per_batch_test is not None:
        cfg.TEST.DETECTIONS_PER_IMAGE = ims_per_batch_test
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thr
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True,
                        help="Path to SparseRCNN yaml (e.g., projects/SparseRCNN/configs/sparsercnn.res50.lite6g.yaml)")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None, help="Model weights .pth")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-proposals", type=int, default=50)
    parser.add_argument("--score-thr", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cfg = build_cfg(args.config_file, num_classes=args.num_classes,
                    num_proposals=args.num_proposals, score_thr=args.score_thr)
    if args.weights:
        cfg.defrost()
        cfg.MODEL.WEIGHTS = args.weights
        cfg.freeze()

    predictor = DefaultPredictor(cfg)

    img = cv2.imread(args.image)
    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]) if len(cfg.DATASETS.TEST) else None, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(args.output, out.get_image()[:, :, ::-1])
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
