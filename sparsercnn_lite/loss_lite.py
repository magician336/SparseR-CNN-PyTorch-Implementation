import torch
import torch.nn.functional as F
from torch import nn

from .util import box_ops
from .util.box_ops import generalized_box_iou
from .util.misc import accuracy, is_dist_avail_and_initialized, get_world_size

def greedy_linear_sum_assignment(cost: torch.Tensor):
    """Greedy fallback for assignment on a 2D cost matrix (cpu tensor).
    Returns two 1D index tensors (rows, cols). Not optimal but sufficient for minimal demo.
    """
    m, n = cost.shape
    cost = cost.clone()
    rows = []
    cols = []
    used_r = set()
    used_c = set()
    k = min(m, n)
    for _ in range(k):
        # find global min among unused rows/cols
        min_val = None
        min_pos = (None, None)
        for i in range(m):
            if i in used_r:
                continue
            # mask used cols by setting large value
            row = cost[i]
            for j in range(n):
                if j in used_c:
                    continue
                v = row[j].item()
                if (min_val is None) or (v < min_val):
                    min_val = v
                    min_pos = (i, j)
        if min_pos[0] is None:
            break
        i, j = min_pos
        used_r.add(i)
        used_c.add(j)
        rows.append(i)
        cols.append(j)
    return torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(cols, dtype=torch.int64)


class SetCriterionLite(nn.Module):
    def __init__(self, cfg, matcher, weight_dict, eos_coef, use_focal):
        super().__init__()
        self.cfg = cfg
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_loss_alpha = cfg.alpha
            self.focal_loss_gamma = cfg.gamma
        else:
            empty_weight = torch.ones(cfg.num_classes)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.cfg.num_classes - 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        if self.use_focal:
            src = src_logits.flatten(0, 1)
            target_classes = target_classes.flatten(0, 1)
            pos_inds = torch.nonzero(target_classes != (self.cfg.num_classes - 1), as_tuple=True)[0]
            labels = torch.zeros_like(src)
            labels[pos_inds, target_classes[pos_inds]] = 1
            prob = src.sigmoid()
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            loss = - (alpha * (1 - prob) ** gamma * labels * (prob + 1e-8).log() +
                      (1 - alpha) * prob ** gamma * (1 - labels) * (1 - prob + 1e-8).log()).sum() / num_boxes
            return {'loss_ce': loss}
        else:
            # Ensure class weight length matches current number of classes
            logits_t = src_logits.transpose(1, 2)
            # logits_t shape: [N, C, Q] where C is num classes
            num_classes = logits_t.shape[1]
            if hasattr(self, 'empty_weight') and self.empty_weight is not None:
                if self.empty_weight.numel() != num_classes:
                    # Resize weights to match number of classes
                    new_weight = torch.ones(num_classes, device=logits_t.device, dtype=logits_t.dtype)
                    # If previous weights exist, copy min(len, num_classes)
                    copy_len = min(self.empty_weight.numel(), num_classes)
                    new_weight[:copy_len] = self.empty_weight[:copy_len].to(new_weight)
                    self.empty_weight = new_weight
                else:
                    self.empty_weight = self.empty_weight.to(logits_t.device, dtype=logits_t.dtype)
            else:
                self.empty_weight = torch.ones(num_classes, device=logits_t.device, dtype=logits_t.dtype)

            loss_ce = F.cross_entropy(logits_t, target_classes, self.empty_weight)
            return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size
        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        if loss == 'labels':
            return self.loss_labels(outputs, targets, indices, num_boxes)
        if loss == 'boxes':
            return self.loss_boxes(outputs, targets, indices, num_boxes)
        raise ValueError(loss)

    def forward(self, outputs, targets):
        outputs_ = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in ['labels', 'boxes']:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux, targets)
                for loss in ['labels', 'boxes']:
                    l_dict = self.get_loss(loss, aux, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class HungarianMatcherLite(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cost_class = cfg.class_weight
        self.cost_bbox = cfg.l1_weight
        self.cost_giou = cfg.giou_weight
        self.use_focal = cfg.use_focal
        if self.use_focal:
            self.focal_loss_alpha = cfg.alpha
            self.focal_loss_gamma = cfg.gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        if self.use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes_xyxy"] for v in targets])
        if self.use_focal:
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]
        else:
            cost_class = -out_prob[:, tgt_ids]
        image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
        image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
        image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        out_bbox_ = out_bbox / image_size_out
        tgt_bbox_ = tgt_bbox / image_size_tgt
        cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            r, cidx = greedy_linear_sum_assignment(c[i])
            indices.append((r, cidx))
        return indices
