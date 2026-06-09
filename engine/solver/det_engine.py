"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable

import torch
import torch.amp
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def _irdino_gradient_stats(model: torch.nn.Module):
    total_sq = 0.0
    active = 0
    count = 0
    for name, parameter in model.named_parameters():
        if not any(tag in name for tag in ('sftb', 'smahe', 'smam', 'ctcbp')) or not parameter.requires_grad:
            continue
        count += 1
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            norm = grad.norm().item()
            total_sq += norm * norm
            if norm > 0:
                active += 1
    return {
        'irdino_grad_norm': total_sq ** 0.5,
        'irdino_grad_active_ratio': active / max(count, 1),
        'irdino_grad_param_tensors': float(count),
    }


def _irdino_forward_stats(outputs):
    diagnostics = outputs.get('irdino_diagnostics', outputs.get('ctcbp_diagnostics', {}))
    return {
        f'irdino_{name}': float(value.detach().float().item())
        for name, value in diagnostics.items()
        if torch.is_tensor(value) and value.numel() == 1
    }




def _tridos_prf1_from_coco_eval(coco_eval, iou_thr=0.5, conf_thr=0.05):
    params = coco_eval.params
    iou_index = int(np.argmin(np.abs(params.iouThrs - iou_thr)))
    area_index = 0
    cat_count = len(params.catIds) if params.useCats else 1
    area_count = len(params.areaRng)
    image_count = len(params.imgIds)
    tp = fp = gt_total = 0
    eval_imgs = coco_eval.evalImgs or []
    for cat_idx in range(cat_count):
        for image_idx in range(image_count):
            idx = cat_idx * area_count * image_count + area_index * image_count + image_idx
            if idx >= len(eval_imgs):
                continue
            entry = eval_imgs[idx]
            if entry is None:
                continue
            gt_ignore = np.asarray(entry.get('gtIgnore', []), dtype=bool)
            gt_total += int((~gt_ignore).sum())
            scores = np.asarray(entry.get('dtScores', []), dtype=float)
            if scores.size == 0:
                continue
            dt_matches = np.asarray(entry.get('dtMatches', []))
            dt_ignore = np.asarray(entry.get('dtIgnore', []), dtype=bool)
            if dt_matches.ndim == 1:
                dt_matches = dt_matches.reshape(1, -1)
            if dt_ignore.ndim == 1:
                dt_ignore = dt_ignore.reshape(1, -1)
            order = np.argsort(-scores, kind='mergesort')
            for det_idx in order:
                if scores[det_idx] < conf_thr:
                    continue
                if det_idx >= dt_ignore.shape[1] or bool(dt_ignore[iou_index, det_idx]):
                    continue
                matched = det_idx < dt_matches.shape[1] and dt_matches[iou_index, det_idx] > 0
                if matched:
                    tp += 1
                else:
                    fp += 1
    fn = max(gt_total - tp, 0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1), int(tp), int(fp), int(fn)


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)
            irdino_forward_stats = _irdino_forward_stats(outputs)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            irdino_gradient_stats = _irdino_gradient_stats(model)
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            irdino_forward_stats = _irdino_forward_stats(outputs)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            irdino_gradient_stats = _irdino_gradient_stats(model)

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(**irdino_forward_stats, **irdino_gradient_stats)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, targets=targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            coco_eval = coco_evaluator.coco_eval['bbox']
            coco_eval_stats = coco_eval.stats.tolist()
            stats['coco_eval_bbox'] = coco_eval_stats

            precisions = coco_eval.eval['precision']
            recalls = coco_eval.eval['recall']
            precision_50 = precisions[0, :, 0, 0, -1]
            recall_50 = recalls[0, 0, 0, -1]
            recall_index = int(np.clip(recall_50 * 100, 0, 100))
            p_approx = float(np.mean(precision_50[:recall_index])) if recall_index > 0 else 0.0
            r_approx = float(recall_50)
            f1 = 2 * p_approx * r_approx / (p_approx + r_approx + 1e-6)

            stats['AP50'] = [float(coco_eval_stats[1])]
            stats['precision'] = [p_approx]
            stats['recall'] = [r_approx]
            stats['f1'] = [float(f1)]
            print(f"\n[COCO Approx Metrics] AP50: {coco_eval_stats[1]:.4f} P: {p_approx:.4f} R: {r_approx:.4f} F1: {f1:.4f}")
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator
