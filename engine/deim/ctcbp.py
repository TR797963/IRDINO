"""
Trainable Counterfactual Target-Clutter Belief adapter.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register


@register()
class CTCBPAdapter(nn.Module):
    """Trainable Module A integrated between the encoder and detector decoder.

    The adapter predicts counterfactual target/clutter evidence on every encoder
    level, applies a small residual feature modulation, and exposes explicit
    explanation/calibration losses to the detector criterion.
    """

    def __init__(
        self,
        in_channels=(224, 224, 224),
        embed_dim=64,
        num_target_atoms=8,
        num_clutter_atoms=16,
        tau=0.25,
        softmin_tau=0.2,
        gate_scale=0.05,
        pos_weight=30.0,
        margin_target=0.05,
        margin_background=0.05,
        margin_clutter=0.05,
        mask_expand=1.5,
        hard_fraction=0.1,
        easy_fraction=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.tau = tau
        self.softmin_tau = softmin_tau
        self.pos_weight = pos_weight
        self.margin_target = margin_target
        self.margin_background = margin_background
        self.margin_clutter = margin_clutter
        self.mask_expand = mask_expand
        self.hard_fraction = hard_fraction
        self.easy_fraction = easy_fraction
        self.eps = eps

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.GroupNorm(self._group_count(embed_dim), embed_dim),
                nn.GELU(),
            )
            for c in in_channels
        ])
        self.bg_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
                nn.GroupNorm(self._group_count(embed_dim), embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            )
            for _ in in_channels
        ])
        self.gate_head = nn.ModuleList([
            nn.Conv2d(4, 1, kernel_size=1)
            for _ in in_channels
        ])
        for head in self.gate_head:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        self.target_atoms = nn.Parameter(torch.randn(num_target_atoms, embed_dim) * 0.02)
        self.clutter_atoms = nn.Parameter(torch.randn(num_clutter_atoms, embed_dim) * 0.02)
        self.residual_gain = nn.Parameter(torch.tensor(float(gate_scale)))

    @staticmethod
    def _group_count(channels: int) -> int:
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    def forward(
        self,
        feats: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        out_feats = []
        level_losses: List[Dict[str, torch.Tensor]] = []
        level_diagnostics: List[Dict[str, torch.Tensor]] = []

        for level, feat in enumerate(feats):
            z = self.proj[level](feat)
            bhat = self.bg_predictor[level](self._masked_context(z))
            e0, e1, ec, likelihood, overlap, uncertainty = self._explain(z, bhat)

            gate_input = torch.cat([likelihood, overlap, uncertainty, e1 - ec], dim=1)
            modulation = torch.tanh(self.gate_head[level](gate_input.float()))
            gain = torch.tanh(self.residual_gain).to(dtype=feat.dtype)
            out_feats.append(feat * (1.0 + gain * modulation.to(dtype=feat.dtype)))

            diagnostics = {
                "modulation_abs": modulation.detach().abs().mean(),
                "likelihood_mean": likelihood.detach().mean(),
                "overlap_mean": overlap.detach().mean(),
                "uncertainty_mean": uncertainty.detach().mean(),
            }

            if self.training and targets is not None:
                target_mask = self._target_mask(
                    targets, feat.shape[-2], feat.shape[-1], feat.device, feat.dtype
                )
                hard_mask, easy_mask = self._background_masks(e0, likelihood, target_mask)
                losses, rule_diagnostics = self._losses(
                    e0, e1, ec, likelihood, uncertainty, target_mask, hard_mask, easy_mask
                )
                level_losses.append(losses)
                diagnostics.update(rule_diagnostics)

            level_diagnostics.append(diagnostics)

        losses = self._mean_dict(level_losses) if level_losses else None
        diagnostics = self._mean_dict(level_diagnostics)
        return out_feats, losses, diagnostics

    def _masked_context(self, z: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1) * 9.0
        return (pooled - z) / 8.0

    def _explain(self, z: torch.Tensor, bhat: torch.Tensor):
        zn = F.normalize(z.float(), dim=1, eps=self.eps)
        bn = F.normalize(bhat.float(), dim=1, eps=self.eps)
        residual = F.normalize(zn - bn, dim=1, eps=self.eps)
        e0 = (zn - bn).pow(2).mean(dim=1, keepdim=True)

        target_atoms = F.normalize(self.target_atoms.float(), dim=1, eps=self.eps)
        target_dot = torch.einsum("bdhw,kd->bkhw", residual, target_atoms)
        target_coef = F.relu(target_dot)
        target_residual = (
            residual.unsqueeze(1)
            - target_coef.unsqueeze(2) * target_atoms.view(1, -1, target_atoms.shape[-1], 1, 1)
        )
        target_dist = target_residual.pow(2).mean(dim=2)
        e1 = self._soft_min(target_dist).unsqueeze(1)

        clutter_atoms = F.normalize(self.clutter_atoms.float(), dim=1, eps=self.eps)
        clutter_dot = torch.einsum("bdhw,kd->bkhw", residual, clutter_atoms)
        clutter_coef = F.relu(clutter_dot)
        clutter_residual = (
            residual.unsqueeze(1)
            - clutter_coef.unsqueeze(2) * clutter_atoms.view(1, -1, clutter_atoms.shape[-1], 1, 1)
        )
        clutter_dist = clutter_residual.pow(2).mean(dim=2)
        ec = self._soft_min(clutter_dist).unsqueeze(1)

        alternative = torch.minimum(e0, ec)
        evidence = alternative - e1
        likelihood = torch.sigmoid(evidence / max(self.tau, self.eps))
        overlap = torch.sigmoid((e0 - e1) / max(self.tau, self.eps))
        uncertainty = torch.exp(-torch.abs(alternative - e1).clamp(max=20.0))
        return e0, e1, ec, likelihood, overlap, uncertainty

    def _soft_min(self, dist: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(-dist / max(self.softmin_tau, self.eps), dim=1)
        return (weights * dist).sum(dim=1)

    def _target_mask(self, targets, height, width, device, dtype):
        mask = torch.zeros((len(targets), 1, height, width), device=device, dtype=dtype)
        for batch_idx, target in enumerate(targets):
            boxes = target.get("boxes")
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.detach()
            cx = boxes[:, 0] * width
            cy = boxes[:, 1] * height
            bw = torch.clamp(boxes[:, 2] * width * self.mask_expand, min=1.0)
            bh = torch.clamp(boxes[:, 3] * height * self.mask_expand, min=1.0)
            x0 = torch.clamp((cx - bw * 0.5).floor().long(), min=0, max=width - 1)
            x1 = torch.clamp((cx + bw * 0.5).ceil().long(), min=0, max=width - 1)
            y0 = torch.clamp((cy - bh * 0.5).floor().long(), min=0, max=height - 1)
            y1 = torch.clamp((cy + bh * 0.5).ceil().long(), min=0, max=height - 1)
            for xa, xb, ya, yb in zip(x0, x1, y0, y1):
                mask[batch_idx, 0, ya:yb + 1, xa:xb + 1] = 1.0
        return mask

    def _background_masks(self, e0, likelihood, target_mask):
        background = target_mask < 0.5
        hard_score = (0.5 * e0.detach() + likelihood.detach()) * background
        easy_score = e0.detach() + likelihood.detach()
        hard_mask = torch.zeros_like(background)
        easy_mask = torch.zeros_like(background)

        for batch_idx in range(background.shape[0]):
            valid = background[batch_idx].flatten().nonzero(as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            hard_k = max(1, int(valid.numel() * self.hard_fraction))
            easy_k = max(1, int(valid.numel() * self.easy_fraction))
            hard_values = hard_score[batch_idx].flatten()[valid]
            easy_values = easy_score[batch_idx].flatten()[valid]
            hard_indices = valid[torch.topk(hard_values, k=min(hard_k, valid.numel())).indices]
            easy_indices = valid[torch.topk(easy_values, k=min(easy_k, valid.numel()), largest=False).indices]
            hard_mask.view(background.shape[0], -1)[batch_idx, hard_indices] = True
            easy_mask.view(background.shape[0], -1)[batch_idx, easy_indices] = True

        return hard_mask, easy_mask

    def _losses(self, e0, e1, ec, likelihood, uncertainty, target_mask, hard_mask, easy_mask):
        target_mask = target_mask.float()
        hard_mask_f = hard_mask.float()
        easy_mask_f = easy_mask.float()
        target_count = target_mask.sum().clamp_min(1.0)
        hard_count = hard_mask_f.sum().clamp_min(1.0)
        easy_count = easy_mask_f.sum().clamp_min(1.0)

        alternative = torch.minimum(e0, ec)
        target_loss = (
            F.relu(self.margin_target + e1 - alternative) * target_mask
        ).sum() / target_count
        background_loss = (
            F.relu(self.margin_background + e0 - e1) * easy_mask_f
        ).sum() / easy_count
        clutter_loss = (
            F.relu(self.margin_clutter + ec - e1) * hard_mask_f
        ).sum() / hard_count

        calibration_mask = (target_mask + hard_mask_f + easy_mask_f).clamp(max=1.0)
        calibration_count = calibration_mask.sum().clamp_min(1.0)
        logits = torch.logit(likelihood.clamp(self.eps, 1.0 - self.eps))
        calibration = F.binary_cross_entropy_with_logits(
            logits,
            target_mask,
            pos_weight=torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype),
            reduction="none",
        )
        calibration_loss = (calibration * calibration_mask).sum() / calibration_count

        uncertainty_loss = (
            (uncertainty * target_mask).sum() / target_count
            + (uncertainty * easy_mask_f).sum() / easy_count
            + ((1.0 - uncertainty) * hard_mask_f).sum() / hard_count
        ) / 3.0

        diagnostics = {
            "target_rule": self._masked_ratio(e1 < alternative, target_mask),
            "background_rule": self._masked_ratio(e0 < e1, easy_mask_f),
            "hard_clutter_rule": self._masked_ratio(ec < e1, hard_mask_f),
            "target_likelihood": self._masked_mean(likelihood, target_mask),
            "hard_likelihood": self._masked_mean(likelihood, hard_mask_f),
            "target_uncertainty": self._masked_mean(uncertainty, target_mask),
            "hard_uncertainty": self._masked_mean(uncertainty, hard_mask_f),
        }
        losses = {
            "loss_ctcbp_target": target_loss,
            "loss_ctcbp_background": background_loss,
            "loss_ctcbp_clutter": clutter_loss,
            "loss_ctcbp_calibration": calibration_loss,
            "loss_ctcbp_uncertainty": uncertainty_loss,
        }
        return losses, diagnostics

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        return (value.detach() * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _masked_ratio(condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        return (condition.detach().float() * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _mean_dict(values: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not values:
            return {}
        return {
            key: torch.stack([item[key] for item in values]).mean()
            for key in values[0]
        }


@register()
class CTCBPTemporalAdapter(nn.Module):
    """Minimal trainable Module B over ordered frames within each rank batch.

    Sequence metadata identifies valid predecessor/current pairs. The adapter
    predicts target-continuation, clutter, and target-birth beliefs, uses them
    to modulate encoder features, and exposes temporal auxiliary losses.
    """

    def __init__(
        self,
        in_channels=(128, 128),
        embed_dim=32,
        max_frame_gap=4,
        gate_scale=0.025,
        margin=0.05,
        mask_expand=1.5,
        eps=1e-6,
    ):
        super().__init__()
        self.max_frame_gap = max_frame_gap
        self.margin = margin
        self.mask_expand = mask_expand
        self.eps = eps
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, embed_dim, kernel_size=1, bias=False),
                nn.GroupNorm(CTCBPAdapter._group_count(embed_dim), embed_dim),
                nn.GELU(),
            )
            for channel in in_channels
        ])
        self.assignment_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(embed_dim, 3, kernel_size=1),
            )
            for _ in in_channels
        ])
        self.state_head = nn.ModuleList([
            nn.Conv2d(embed_dim, 1, kernel_size=1)
            for _ in in_channels
        ])
        self.residual_gain = nn.Parameter(torch.tensor(float(gate_scale)))

    def forward(
        self,
        feats: List[torch.Tensor],
        targets: Optional[List[dict]] = None,
    ) -> Tuple[List[torch.Tensor], Optional[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if targets is None or not targets or 'sequence_id' not in targets[0]:
            zero = feats[0].sum() * 0.0
            return feats, None, {
                'pair_ratio': zero.detach(),
                'modulation_abs': zero.detach(),
            }

        device = feats[0].device
        sequence_ids = torch.cat([target['sequence_id'].reshape(-1) for target in targets]).to(device)
        frame_indices = torch.cat([target['frame_index'].reshape(-1) for target in targets]).to(device)
        previous_sequence_ids = sequence_ids.roll(1)
        previous_frame_indices = frame_indices.roll(1)
        frame_gap = frame_indices - previous_frame_indices
        valid_pair = (
            (sequence_ids == previous_sequence_ids)
            & (frame_gap > 0)
            & (frame_gap <= self.max_frame_gap)
        )
        valid_pair[0] = False

        out_feats = []
        level_losses: List[Dict[str, torch.Tensor]] = []
        level_diagnostics: List[Dict[str, torch.Tensor]] = []
        for level, feat in enumerate(feats):
            descriptor = self.proj[level](feat).float()
            previous = descriptor.roll(1, dims=0)
            pair_features = torch.cat([descriptor, previous, (descriptor - previous).abs()], dim=1)
            assignment_logits = self.assignment_head[level](pair_features)
            assignment = assignment_logits.softmax(dim=1)
            state = self.state_head[level](descriptor).sigmoid()

            valid_map = valid_pair.view(-1, 1, 1, 1).expand(
                -1, 1, feat.shape[-2], feat.shape[-1]
            )
            if self.training:
                target_mask = CTCBPAdapter._target_mask(
                    self, targets, feat.shape[-2], feat.shape[-1], feat.device, feat.dtype
                ) > 0.5
            else:
                target_mask = torch.zeros_like(valid_map)
            previous_target_mask = target_mask.roll(1, dims=0)
            continuation = valid_map & target_mask & previous_target_mask
            target_birth = valid_map & target_mask & ~previous_target_mask
            clutter = valid_map & ~target_mask
            labels = torch.ones(
                (feat.shape[0], feat.shape[-2], feat.shape[-1]),
                dtype=torch.long,
                device=device,
            )
            labels[continuation.squeeze(1)] = 0
            labels[target_birth.squeeze(1)] = 2

            temporal_score = assignment[:, 0] + assignment[:, 2] - assignment[:, 1]
            temporal_score = temporal_score.unsqueeze(1) * valid_map.float()
            gain = torch.tanh(self.residual_gain).to(dtype=feat.dtype)
            modulation = temporal_score.to(dtype=feat.dtype)
            out_feats.append(feat * (1.0 + gain * modulation))

            assignment_class = assignment.argmax(dim=1)
            level_diagnostics.append({
                'pair_ratio': valid_pair.float().mean().detach(),
                'modulation_abs': modulation.abs().mean().detach(),
                'pi_target_mean': self._masked_mean(
                    assignment[:, 0] + assignment[:, 2], valid_map.squeeze(1)
                ),
                'kappa_clutter_mean': self._masked_mean(assignment[:, 1], valid_map.squeeze(1)),
                'repeated_false_to_clutter': self._masked_ratio(
                    assignment_class == 1, clutter.squeeze(1)
                ),
                'true_target_to_target': self._masked_ratio(
                    assignment_class != 1, (continuation | target_birth).squeeze(1)
                ),
                'true_target_to_clutter': self._masked_ratio(
                    assignment_class == 1, (continuation | target_birth).squeeze(1)
                ),
                'background_target_birth': self._masked_ratio(
                    assignment_class == 2, clutter.squeeze(1)
                ),
            })

            if self.training:
                zero = assignment_logits.sum() * 0.0
                valid_pixels = valid_map.squeeze(1)
                if valid_pixels.any():
                    assignment_loss_map = F.cross_entropy(
                        assignment_logits,
                        labels,
                        reduction='none',
                        weight=assignment_logits.new_tensor([5.0, 1.0, 5.0]),
                    )
                    assignment_loss = self._masked_loss(
                        assignment_loss_map,
                        valid_pixels,
                        zero,
                    )
                else:
                    assignment_loss = zero
                same_state = valid_map & (target_mask == previous_target_mask)
                consistency_loss = self._masked_loss(
                    (state - state.roll(1, dims=0)).pow(2),
                    same_state,
                    zero,
                )
                clutter_suppression_loss = self._masked_loss(
                    F.relu(
                        self.margin + assignment[:, 0] + assignment[:, 2] - assignment[:, 1]
                    ),
                    clutter.squeeze(1),
                    zero,
                )
                target_protection_loss = self._masked_loss(
                    F.relu(
                        self.margin + assignment[:, 1] - assignment[:, 0] - assignment[:, 2]
                    ),
                    (continuation | target_birth).squeeze(1),
                    zero,
                )
                level_losses.append({
                    'loss_ctcbp_temporal_assignment': assignment_loss,
                    'loss_ctcbp_temporal_consistency': consistency_loss,
                    'loss_ctcbp_temporal_clutter_suppression': clutter_suppression_loss,
                    'loss_ctcbp_temporal_target_protection': target_protection_loss,
                })

        losses = CTCBPAdapter._mean_dict(level_losses) if level_losses else None
        diagnostics = CTCBPAdapter._mean_dict(level_diagnostics)
        return out_feats, losses, diagnostics

    @staticmethod
    def _masked_loss(value: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=value.dtype)
        return (value * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value.detach() * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

    @staticmethod
    def _masked_ratio(condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (condition.detach().float() * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
