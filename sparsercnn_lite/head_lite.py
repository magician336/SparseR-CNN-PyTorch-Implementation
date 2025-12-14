import math
import copy
from typing import List, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import roi_align


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class DynamicHeadLite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        # Convention in this repo:
        # - cfg.num_classes is TOTAL classes including the last "no-object/background" index
        # - softmax branch predicts all classes including background (C = cfg.num_classes)
        # - focal branch predicts only foreground classes (C = cfg.num_classes - 1)
        num_classes = (cfg.num_classes - 1) if cfg.use_focal else cfg.num_classes

        dim_feedforward = cfg.dim_feedforward
        nhead = cfg.nheads
        dropout = cfg.dropout
        activation = cfg.activation
        num_heads = cfg.num_heads

        rcnn_head = RCNNHeadLite(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = nn.ModuleList([copy.deepcopy(rcnn_head) for _ in range(num_heads)])
        self.return_intermediate = cfg.deep_supervision

        self.use_focal = cfg.use_focal
        self.num_classes = cfg.num_classes
        if self.use_focal:
            prior_prob = cfg.prior_prob
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters(num_classes)

        self.pooler_resolution = cfg.pooler_resolution
        self.feature_stride = cfg.feature_stride

    def _reset_parameters(self, num_classes: int):
        # Default Xavier init for weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # For focal classification, set classification bias to prior prob
        if self.use_focal:
            for head in self.head_series:
                if hasattr(head, "class_logits") and head.class_logits.bias is not None and head.class_logits.bias.shape[0] == num_classes:
                    nn.init.constant_(head.class_logits.bias, self.bias_value)

    def _roi_pool(self, feat: Tensor, boxes: Tensor) -> Tensor:
        # boxes: (N, nr_boxes, 4) absolute in pixels
        N, nr_boxes, _ = boxes.shape
        boxes_with_idx = []
        for b in range(N):
            bboxes = boxes[b]
            idx = torch.full((bboxes.size(0), 1), b, device=bboxes.device, dtype=bboxes.dtype)
            boxes_with_idx.append(torch.cat([idx, bboxes], dim=1))
        rois = torch.cat(boxes_with_idx, dim=0)
        # spatial_scale = 1/stride
        pooled = roi_align(feat, rois, output_size=self.pooler_resolution, spatial_scale=1.0 / self.feature_stride, aligned=True)
        # Output: (N*nr_boxes, C, H, W) -> reshape to (H*W, N*nr_boxes, C) to match original head
        Ntotal = N * nr_boxes
        pooled = pooled.view(Ntotal, feat.shape[1], -1).permute(2, 0, 1)
        return pooled

    def forward(self, feature: Tensor, init_bboxes: Tensor, init_features: Tensor):
        inter_class_logits = []
        inter_pred_bboxes = []

        bs = feature.shape[0]
        bboxes = init_bboxes
        proposal_features = init_features[None].repeat(1, bs, 1).clone()

        for rcnn_head in self.head_series:
            class_logits, pred_bboxes, proposal_features = rcnn_head(feature, bboxes, proposal_features, self._roi_pool)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)
        return class_logits[None], pred_bboxes[None]


class RCNNHeadLite(nn.Module):
    def __init__(self, cfg, d_model, num_classes, dim_feedforward=1024, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConvLite(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        num_cls = cfg.num_cls
        cls_module = []
        for _ in range(num_cls):
            cls_module += [nn.Linear(d_model, d_model, False), nn.LayerNorm(d_model), nn.ReLU(inplace=True)]
        self.cls_module = nn.Sequential(*cls_module)

        num_reg = cfg.num_reg
        reg_module = []
        for _ in range(num_reg):
            reg_module += [nn.Linear(d_model, d_model, False), nn.LayerNorm(d_model), nn.ReLU(inplace=True)]
        self.reg_module = nn.Sequential(*reg_module)

        self.use_focal = cfg.use_focal
        # Use provided num_classes: for softmax, it should include background; for focal, exact number of foreground classes
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, feature: Tensor, bboxes: Tensor, pro_features: Tensor, pooler):
        N, nr_boxes = bboxes.shape[:2]
        roi_features = pooler(feature, bboxes)  # (H*W, N*nr_boxes, C)

        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = self.cls_module(fc_feature.clone())
        reg_feature = self.reg_module(fc_feature.clone())
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        boxes = boxes.to(deltas.dtype)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
        return pred_boxes


class DynamicConvLite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.dim_dynamic = 64
        self.num_dynamic = 2
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)
        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        num_output = self.hidden_dim * (cfg.pooler_resolution ** 2)
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        # pro_features: (1, N*nr_boxes, d_model)
        # roi_features: (H*W, N*nr_boxes, d_model)
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)
        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)
        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)
        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)
        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)
        return features


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
