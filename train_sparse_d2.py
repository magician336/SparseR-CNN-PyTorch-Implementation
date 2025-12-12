import argparse
import os
import sys

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

# Register project module for SparseRCNN
PROJ_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "projects", "SparseRCNN")
if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)
from sparsercnn import add_sparsercnn_config  # noqa: F401

# Import personal dataset helper
sys.path.append(os.path.dirname(__file__))
from data_synth import SynthRectDataset  # noqa: E402


def register_personal_datasets(data_root: str):
    def _make(split):
        ds = SynthRectDataset(data_root, split=split)
        return ds.to_coco_format()

    DatasetCatalog.register("personal_train", lambda: _make("train"))
    DatasetCatalog.register("personal_val", lambda: _make("val"))
    MetadataCatalog.get("personal_train").set(thing_classes=["rect"])
    MetadataCatalog.get("personal_val").set(thing_classes=["rect"])


def setup_cfg(args):
    cfg = get_cfg()
    add_sparsercnn_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Datasets
    cfg.DATASETS.TRAIN = ("personal_train",)
    cfg.DATASETS.TEST = ("personal_val",)

    # Output
    cfg.OUTPUT_DIR = args.output_dir

    # VRAM constraints and overrides
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.INPUT.MIN_SIZE_TRAIN = (args.min_size_train,)
    cfg.MODEL.SparseRCNN.NUM_PROPOSALS = args.num_proposals
    cfg.MODEL.SparseRCNN.NUM_CLASSES = args.num_classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes

    # Eval period for periodic validation
    cfg.TEST.EVAL_PERIOD = args.eval_period

    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="personal/data")
    parser.add_argument("--config-file", type=str, default="projects/SparseRCNN/configs/sparsercnn.res50.lite6g.yaml")
    parser.add_argument("--output-dir", type=str, default="personal/outputs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-size-train", type=int, default=800)
    parser.add_argument("--num-proposals", type=int, default=50)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-period", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Register datasets
    register_personal_datasets(args.data_root)

    # Build config
    cfg = setup_cfg(args)

    # Custom Trainer with COCO evaluator
    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    # Eval-only path
    if args.eval_only:
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=True)
        results = trainer.test(cfg, trainer.model)
        print("Evaluation results:", results)
        return

    # Trainer
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print(f"Training done. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
