from .detector_lite import SparseRCNNLite, default_sparsercnn_cfg

# 仅在检测到 Detectron2 时尝试注册，避免强制依赖
try:
    from detectron2.modeling import META_ARCH_REGISTRY
    META_ARCH_REGISTRY.register(SparseRCNNLite)
except ImportError:
    pass
