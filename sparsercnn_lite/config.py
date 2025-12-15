from dataclasses import dataclass


@dataclass
class SparseRCNNLiteCfg:
    # Model
    num_classes: int = 2  # include background? head handles background internally when use_focal=False
    num_proposals: int = 100
    hidden_dim: int = 256
    num_heads: int = 3
    use_focal: bool = False
    prior_prob: float = 0.01
    alpha: float = 0.25
    gamma: float = 2.0
    # RCNN head
    dim_feedforward: int = 1024
    nheads: int = 8
    dropout: float = 0.1
    activation: str = "relu"
    num_cls: int = 3
    num_reg: int = 3
    # Loss weights
    class_weight: float = 2.0
    l1_weight: float = 5.0
    giou_weight: float = 2.0
    no_object_weight: float = 0.2
    deep_supervision: bool = True
    # Input
    pixel_mean: tuple = (123.675, 116.28, 103.53)  # ImageNet
    pixel_std: tuple = (58.395, 57.12, 57.375)
    size_divisibility: int = 16
    # ROI Align
    pooler_resolution: int = 7
    feature_stride: int = 16  # use a single stride-16 feature map
