import torch


def accuracy(x, y):
    """Compute top-1 accuracy for logits x and target y."""
    if x.numel() == 0:
        return torch.tensor(0.0),
    preds = x.argmax(dim=-1)
    correct = (preds == y).float().sum()
    return (100.0 * correct / (y.numel() + 1e-7),)


def is_dist_avail_and_initialized():
    return False


def get_world_size():
    return 1
