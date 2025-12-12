import argparse
import os
import sys

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

sys.path.append(os.path.dirname(__file__))
from data_synth import SynthRectDataset


def register_datasets(data_root):
    # Register synthetic datasets
    DatasetCatalog.register("personal_train", lambda: SynthRectDataset(data_root, split="train").to_coco_format())
    DatasetCatalog.register("personal_val", lambda: SynthRectDataset(data_root, split="val").to_coco_format())
    MetadataCatalog.get("personal_train").set(thing_classes=["rect"])
    MetadataCatalog.get("personal_val").set(thing_classes=["rect"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="personal/data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output-dir", type=str, default="personal/outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Register datasets
    register_datasets(args.data_root)

    # Load config
    cfg = get_cfg()
    cfg.merge_from_file("personal/config_lite.yaml")
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.epochs * 100  # Approximate epochs to iterations

    # Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print(f"Training completed. Model saved in {args.output_dir}")


if __name__ == "__main__":
    main()
