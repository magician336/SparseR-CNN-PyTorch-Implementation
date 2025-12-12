import math
from typing import List, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from .config import SparseRCNNLiteCfg
from .head_lite import DynamicHeadLite
from .loss_lite import SetCriterionLite, HungarianMatcherLite
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def default_sparsercnn_cfg() -> SparseRCNNLiteCfg:
    return SparseRCNNLiteCfg()


class BackboneFPNLite(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        try:
            backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = torchvision.models.resnet50(pretrained=True)
        # Extract layer1, layer2, layer3 feature maps, choose layer3 (stride=16)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # stride 4
        self.layer2 = backbone.layer2  # stride 8
        self.layer3 = backbone.layer3  # stride 16
        self.out_conv = nn.Conv2d(1024, out_channels, kernel_size=1)
        self.stride = 16

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        y = self.out_conv(x)
        return y  # (N, C, H/16, W/16)


class SparseRCNNLite(nn.Module):
    def __init__(self, cfg: SparseRCNNLiteCfg = None):
        super().__init__()
        self.cfg = cfg or default_sparsercnn_cfg()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pixel_mean = torch.tensor(self.cfg.pixel_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(self.cfg.pixel_std, dtype=torch.float32).view(1, 3, 1, 1)

        self.backbone = BackboneFPNLite(out_channels=self.cfg.hidden_dim)
        self.size_divisibility = self.cfg.size_divisibility

        self.num_classes = self.cfg.num_classes
        self.num_proposals = self.cfg.num_proposals
        self.hidden_dim = self.cfg.hidden_dim

        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim)
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)

        self.head = DynamicHeadLite(self.cfg)

        matcher = HungarianMatcherLite(self.cfg)
        weight_dict = {"loss_ce": self.cfg.class_weight, "loss_bbox": self.cfg.l1_weight, "loss_giou": self.cfg.giou_weight}
        self.criterion = SetCriterionLite(cfg=self.cfg, matcher=matcher, weight_dict=weight_dict,
                                          eos_coef=self.cfg.no_object_weight, use_focal=self.cfg.use_focal)
        self.to(self.device)

    def normalize(self, x):
        mean = self.pixel_mean.to(x.device)
        std = self.pixel_std.to(x.device)
        return (x * 255.0 - mean) / std

    def _pad_to_stride(self, x: torch.Tensor, stride: int):
        h, w = x.shape[-2:]
        nh = (h + stride - 1) // stride * stride
        nw = (w + stride - 1) // stride * stride
        pad_h = nh - h
        pad_w = nw - w
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]] = None):
        # images: list of [C,H,W] in [0,1]
        device = self.device
        images = [img.to(device) for img in images]
        sizes = [img.shape[-2:] for img in images]
        images_whwh = torch.stack([torch.tensor([w, h, w, h], dtype=torch.float32, device=device) for h, w in sizes], dim=0)
        x = torch.stack([self._pad_to_stride(self.normalize(img.unsqueeze(0)), self.size_divisibility).squeeze(0) for img in images], dim=0)
        feat = self.backbone(x)

        proposal_boxes = self.init_proposal_boxes.weight.clone()
        proposal_boxes = box_cxcywh_to_xyxy(proposal_boxes)
        proposal_boxes = proposal_boxes[None] * images_whwh[:, None, :]

        outputs_class, outputs_coord = self.head(feat, proposal_boxes, self.init_proposal_features.weight)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.training and targets is not None:
            new_targets = []
            for i, t in enumerate(targets):
                h, w = sizes[i]
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float32, device=device)
                gt_boxes = t['boxes']  # xyxy absolute
                target = {
                    'labels': t['labels'].to(device),
                    'boxes': box_xyxy_to_cxcywh(gt_boxes / image_size_xyxy),
                    'boxes_xyxy': gt_boxes.to(device),
                    'image_size_xyxy': image_size_xyxy,
                    'image_size_xyxy_tgt': image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
                }
                new_targets.append(target)
            loss_dict = self.criterion(output, new_targets)
            for k in list(loss_dict.keys()):
                if k in self.criterion.weight_dict:
                    loss_dict[k] *= self.criterion.weight_dict[k]
            return loss_dict
        else:
            box_cls = output['pred_logits']
            box_pred = output['pred_boxes']
            if self.cfg.use_focal:
                scores = box_cls.sigmoid()
                labels = torch.arange(self.num_classes, device=device).unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
                results = []
                for i in range(scores.shape[0]):
                    s = scores[i].flatten(0, 1)
                    topk_scores, topk_idx = s.topk(self.num_proposals, sorted=False)
                    labels_i = labels[topk_idx]
                    box_i = box_pred[i].view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)[topk_idx]
                    results.append({'boxes': box_i, 'scores': topk_scores, 'labels': labels_i})
                return results
            else:
                scores, labels = F.softmax(box_cls, dim=-1).max(-1)
                results = []
                for i in range(scores.shape[0]):
                    results.append({'boxes': box_pred[i], 'scores': scores[i], 'labels': labels[i]})
                return results
