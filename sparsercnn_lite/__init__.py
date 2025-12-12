from .detector_lite import SparseRCNNLite, default_sparsercnn_cfg

# Register the model with detectron2
from detectron2.modeling import META_ARCH_REGISTRY
META_ARCH_REGISTRY.register(SparseRCNNLite)
