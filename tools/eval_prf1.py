#!/usr/bin/env python3
"""Compute Precision/Recall/F1 for COCO-style detections at IoU=0.5."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def xywh_to_xyxy(box):
    x, y, w, h = box
    return x, y, x + w, y + h


def iou(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_ground_truth(path: Path):
    data = json.loads(path.read_text())
    by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        by_image[ann["image_id"]].append(ann)
    return by_image


def load_predictions(path: Path, conf: float):
    data = json.loads(path.read_text())
    by_image = defaultdict(list)
    for pred in data:
        if pred.get("score", 1.0) >= conf:
            by_image[pred["image_id"]].append(pred)
    for preds in by_image.values():
        preds.sort(key=lambda item: item.get("score", 1.0), reverse=True)
    return by_image


def compute(gt_by_image, pred_by_image, iou_thr: float):
    tp = fp = 0
    total_gt = sum(len(items) for items in gt_by_image.values())
    image_ids = set(gt_by_image) | set(pred_by_image)
    for image_id in image_ids:
        gt_items = gt_by_image.get(image_id, [])
        pred_items = pred_by_image.get(image_id, [])
        matched = set()
        for pred in pred_items:
            best_idx, best_iou = None, 0.0
            for idx, gt in enumerate(gt_items):
                if idx in matched:
                    continue
                if pred.get("category_id") != gt.get("category_id"):
                    continue
                score = iou(pred["bbox"], gt["bbox"])
                if score > best_iou:
                    best_idx, best_iou = idx, score
            if best_idx is not None and best_iou >= iou_thr:
                matched.add(best_idx)
                tp += 1
            else:
                fp += 1
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="COCO ground-truth annotation JSON")
    parser.add_argument("--pred", required=True, help="COCO detection result JSON")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()

    result = compute(
        load_ground_truth(Path(args.gt)),
        load_predictions(Path(args.pred), args.conf),
        args.iou,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
