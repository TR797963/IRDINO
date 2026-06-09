"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        best_metrics = {}
        selected_best = {
            'epoch': -1,
            'ap50': float('-inf'),
            'f1': float('-inf'),
            'recall': float('-inf'),
            'precision': float('-inf'),
        }
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            resume_metrics = {}
            if 'AP50' in test_stats:
                resume_metrics['ap50'] = float(test_stats['AP50'][0])
            elif 'coco_eval_bbox' in test_stats and len(test_stats['coco_eval_bbox']) > 1:
                resume_metrics['ap50'] = float(test_stats['coco_eval_bbox'][1])
            if 'f1' in test_stats:
                resume_metrics['f1'] = float(test_stats['f1'][0])
            if 'recall' in test_stats:
                resume_metrics['recall'] = float(test_stats['recall'][0])
            if 'precision' in test_stats:
                resume_metrics['precision'] = float(test_stats['precision'][0])
            selected_best.update(resume_metrics)
            selected_best['epoch'] = self.last_epoch
            print(
                "Resuming with selected best: "
                f"AP50={selected_best['ap50']:.4f}, F1={selected_best['f1']:.4f}, "
                f"Recall={selected_best['recall']:.4f}, Precision={selected_best['precision']:.4f}"
            )

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            current_ap50 = 0.0
            if 'AP50' in test_stats:
                current_ap50 = float(test_stats['AP50'][0])
            elif 'coco_eval_bbox' in test_stats and len(test_stats['coco_eval_bbox']) > 1:
                current_ap50 = float(test_stats['coco_eval_bbox'][1])

            current_f1 = 0.0
            if 'f1' in test_stats:
                current_f1 = float(test_stats['f1'][0])

            current_recall = 0.0
            if 'recall' in test_stats:
                current_recall = float(test_stats['recall'][0])

            current_precision = 0.0
            if 'precision' in test_stats:
                current_precision = float(test_stats['precision'][0])

            metric_values = {}
            if 'coco_eval_bbox' in test_stats:
                metric_values['ap'] = float(test_stats['coco_eval_bbox'][0])
            for metric_name in ['AP50', 'precision', 'recall', 'f1']:
                if metric_name in test_stats:
                    metric_values[metric_name.lower()] = float(test_stats[metric_name][0])

            if self.output_dir and metric_values:
                eval_dir = self.output_dir / 'eval'
                eval_dir.mkdir(exist_ok=True)
                for metric_name, metric_value in metric_values.items():
                    previous = best_metrics.get(metric_name, {'value': float('-inf')})
                    if metric_value > previous['value']:
                        best_metrics[metric_name] = {'epoch': epoch, 'value': metric_value}
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / f'best_{metric_name}.pth')
                        if dist_utils.is_main_process() and coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                eval_dir / f'best_{metric_name}.pth'
                            )

                if dist_utils.is_main_process():
                    with (self.output_dir / 'best_metrics.json').open('w') as f:
                        json.dump(best_metrics, f, indent=2)

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                best_stat_print[k] = best_stat[k]

            if dist_utils.is_main_process():
                print(f'best_stat: {best_stat_print}')

            current_selected = {
                'epoch': epoch,
                'ap50': current_ap50,
                'f1': current_f1,
                'recall': current_recall,
                'precision': current_precision,
            }
            current_key = (
                current_selected['ap50'],
                current_selected['f1'],
                current_selected['recall'],
                current_selected['precision'],
                current_selected['epoch'],
            )
            best_key = (
                selected_best['ap50'],
                selected_best['f1'],
                selected_best['recall'],
                selected_best['precision'],
                selected_best['epoch'],
            )
            is_best_model = current_key > best_key

            print(
                f"Epoch {epoch} Result: AP50={current_ap50:.4f}, F1={current_f1:.4f}, "
                f"Recall={current_recall:.4f}, Precision={current_precision:.4f}"
            )
            print(
                "Current Best:       "
                f"AP50={selected_best['ap50']:.4f}, F1={selected_best['f1']:.4f}, "
                f"Recall={selected_best['recall']:.4f}, Precision={selected_best['precision']:.4f}"
            )

            if is_best_model and self.output_dir:
                previous_best = selected_best.copy()
                selected_best = current_selected
                print(
                    "--> Improved selected best by priority "
                    "(AP50 > F1 > Recall > Precision > latest epoch): "
                    f"({previous_best['ap50']:.4f}, {previous_best['f1']:.4f}, "
                    f"{previous_best['recall']:.4f}, {previous_best['precision']:.4f}) -> "
                    f"({selected_best['ap50']:.4f}, {selected_best['f1']:.4f}, "
                    f"{selected_best['recall']:.4f}, {selected_best['precision']:.4f})"
                )
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_selected.pth')
                dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_f1.pth')
                if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                else:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')
                if dist_utils.is_main_process() and coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                    eval_dir = self.output_dir / 'eval'
                    eval_dir.mkdir(exist_ok=True)
                    torch.save(coco_evaluator.coco_eval["bbox"].eval, eval_dir / 'best_selected.pth')
                    with (self.output_dir / 'selected_best.json').open('w') as f:
                        json.dump(selected_best, f, indent=2)
            elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                best_stat = {'epoch': -1, }
                self.ema.decay -= 0.0001
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters,
                'best_ap50': selected_best['ap50'],
                'best_f1': selected_best['f1'],
                'best_recall': selected_best['recall'],
                'best_precision': selected_best['precision'],
                'best_selected_epoch': selected_best['epoch'],
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
