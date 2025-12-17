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
        """
        构建多层 RCNN 头（迭代 refinement）。
        参数：
            cfg: SparseRCNNLiteCfg 配置，含通道数/分类数/迭代次数等。
        返回：无；初始化 head_series、ROI 池化配置与参数初始化。
        """
        super().__init__()
        self.cfg = cfg
        d_model = cfg.hidden_dim
        # 约定：use_focal=False 时 num_classes 含背景；use_focal=True 时只包含前景类
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
        """
        初始化参数：权重 Xavier，focal 模式下分类 bias 按 prior_prob 设置。
        参数：num_classes - 分类头输出维度（与 focal/softmax 分支一致）。
        返回：无。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.use_focal:
            for head in self.head_series:
                if hasattr(head, "class_logits") and head.class_logits.bias is not None and head.class_logits.bias.shape[0] == num_classes:
                    # focal 分类头按 prior_prob 初始化，避免初始正例概率过低
                    nn.init.constant_(head.class_logits.bias, self.bias_value)

    def _roi_pool(self, feat: Tensor, boxes: Tensor) -> Tensor:
        """
        对给定 proposals 做 ROIAlign。
        参数：
            feat: 主干特征图 (N, C, H, W)。
            boxes: proposals 绝对坐标 (N, nr_boxes, 4)。
        返回：
            pooled: (H*W, N*nr_boxes, C) 形状的池化结果，供动态卷积使用。
        """
        # 将 (N, nr_boxes, 4) 绝对坐标打包成 ROIAlign 需要的 (idx, x1, y1, x2, y2)
        N, nr_boxes, _ = boxes.shape
        boxes_with_idx = []
        for b in range(N):
            bboxes = boxes[b]
            idx = torch.full((bboxes.size(0), 1), b, device=bboxes.device, dtype=bboxes.dtype)
            boxes_with_idx.append(torch.cat([idx, bboxes], dim=1))
        rois = torch.cat(boxes_with_idx, dim=0)
        pooled = roi_align(feat, rois, output_size=self.pooler_resolution, spatial_scale=1.0 / self.feature_stride, aligned=True)
        Ntotal = N * nr_boxes
        pooled = pooled.view(Ntotal, feat.shape[1], -1).permute(2, 0, 1)
        return pooled

    def forward(self, feature: Tensor, init_bboxes: Tensor, init_features: Tensor):
        """
        逐层 refinement：依次调用多头 RCNNHeadLite，更新 proposals 与特征。
        参数：
            feature: 主干特征图 (N, C, H, W)。
            init_bboxes: 初始 proposals (N, Q, 4) 绝对坐标。
            init_features: 初始 proposal 特征 (Q, C)。
        返回：
            若 deep_supervision=True，返回 (cls_logits_stack, bbox_stack)；否则返回末层的 (cls_logits, bbox)。形状：
                cls_logits: (num_heads, N, Q, num_classes) 或 (1, N, Q, num_classes)
                bbox: (num_heads, N, Q, 4) 或 (1, N, Q, 4)
        """
        # 按迭代次数串联多个 RCNNHeadLite：每一轮更新 proposal boxes 与特征
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
        """
        单层 RCNN 头：自注意力 + 动态卷积交互 + FFN，输出分类和 bbox 偏移。
        参数：
            cfg: 配置；d_model: 特征维度；num_classes: 分类输出维度；
            dim_feedforward/nhead/dropout/activation: Transformer FFN/注意力超参；
            scale_clamp/bbox_weights: bbox 解码时的尺度限制与缩放权重。
        返回：无（构造模块）。
        """
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
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, feature: Tensor, bboxes: Tensor, pro_features: Tensor, pooler):
        """
        执行一次 head 迭代：ROI 池化 -> 自注意力 -> 动态卷积 -> FFN -> 分类/回归。
        参数：
            feature: 主干特征 (N, C, H, W)
            bboxes: 当前 proposals (N, Q, 4)
            pro_features: 当前 proposal 特征 (1, Q, C)
            pooler: ROIAlign 函数
        返回：
            class_logits: (N, Q, num_classes)
            pred_bboxes: (N, Q, 4) 解码后的绝对坐标
            obj_features: (1, N*Q, C) 下一层的 proposal 特征
        """
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
        """
        将 (dx, dy, dw, dh) 偏移解码为 xyxy 绝对坐标。
        参数：
            deltas: 回归输出 (N*Q, 4)
            boxes: 当前 boxes (N*Q, 4) 作为参考框
        返回：
            pred_boxes: 解码后的 xyxy (N*Q, 4)
        """
        # 将 (dx,dy,dw,dh) 变换回 xyxy 绝对坐标，Clamp 防止尺度爆炸
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
        """
        动态卷积模块：根据 proposal 特征生成两层 1x1 卷积权重，对 ROI 特征自适应变换。
        参数：cfg 提供 hidden_dim 和 pooler_resolution。
        返回：无。
        """
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
        # 动态生成两层 1x1 卷积权重，对 ROI 特征做自适应变换
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
    """
    返回指定名称的激活函数。
    参数：activation in {"relu","gelu","glu"}
    返回：对应的 torch 函数句柄。
    """
    # 简单激活函数工厂
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
