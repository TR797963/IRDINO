# Open Source Check

Open-source directory: IRDINO-open

## Smoke Tests

- Python compile: PASS
- Train help: PASS
- Eval help: PASS
- Dataset checker help: PASS
- PRF1 helper help: PASS

## Safety Checks

- Large file check: PASS
- Weight file check: PASS
- Secret filename check: PASS
- Absolute path check: PASS
- Old internal project reference check: PASS
- Generated cache check: PASS
- Data/result directory check: PASS

## Files Included

- `.gitignore`
- `LICENSE`
- `OPEN_SOURCE_CHECK.md`
- `README.md`
- `configs/base/dataloader.yml`
- `configs/base/deimv2.yml`
- `configs/base/optimizer.yml`
- `configs/dataset_irstd/daub_detection.yml`
- `configs/dataset_irstd/irdst_detection.yml`
- `configs/dataset_irstd/itsdt_15k_detection.yml`
- `configs/irdino/irdino_smahe_l_daub.yml`
- `configs/irdino/irdino_smahe_l_irdst.yml`
- `configs/irdino/irdino_smahe_l_itsdt15k.yml`
- `configs/irdino/irdino_smahe_n_daub.yml`
- `configs/irdino/irdino_smahe_n_irdst.yml`
- `configs/irdino/irdino_smahe_n_itsdt15k.yml`
- `configs/irstd/deimv2_dinov3_l_irstd_base.yml`
- `configs/irstd/deimv2_hgnetv2_n_irstd_base.yml`
- `configs/runtime.yml`
- `docs/DATASET.md`
- `docs/MODEL_ZOO.md`
- `engine/__init__.py`
- `engine/backbone/__init__.py`
- `engine/backbone/common.py`
- `engine/backbone/csp_darknet.py`
- `engine/backbone/csp_resnet.py`
- `engine/backbone/dinov3/__init__.py`
- `engine/backbone/dinov3/layers/__init__.py`
- `engine/backbone/dinov3/layers/attention.py`
- `engine/backbone/dinov3/layers/block.py`
- `engine/backbone/dinov3/layers/dino_head.py`
- `engine/backbone/dinov3/layers/ffn_layers.py`
- `engine/backbone/dinov3/layers/fp8_linear.py`
- `engine/backbone/dinov3/layers/layer_scale.py`
- `engine/backbone/dinov3/layers/patch_embed.py`
- `engine/backbone/dinov3/layers/rms_norm.py`
- `engine/backbone/dinov3/layers/rope_position_encoding.py`
- `engine/backbone/dinov3/layers/sparse_linear.py`
- `engine/backbone/dinov3/utils/__init__.py`
- `engine/backbone/dinov3/utils/cluster.py`
- `engine/backbone/dinov3/utils/custom_callable.py`
- `engine/backbone/dinov3/utils/dtype.py`
- `engine/backbone/dinov3/utils/utils.py`
- `engine/backbone/dinov3/vision_transformer.py`
- `engine/backbone/dinov3_adapter.py`
- `engine/backbone/hgnetv2.py`
- `engine/backbone/ms_deform_attn.py`
- `engine/backbone/presnet.py`
- `engine/backbone/test_resnet.py`
- `engine/backbone/timm_model.py`
- `engine/backbone/torchvision_model.py`
- `engine/backbone/utils.py`
- `engine/backbone/vit_tiny.py`
- `engine/core/__init__.py`
- `engine/core/_config.py`
- `engine/core/workspace.py`
- `engine/core/yaml_config.py`
- `engine/core/yaml_utils.py`
- `engine/data/__init__.py`
- `engine/data/_misc.py`
- `engine/data/dataloader.py`
- `engine/data/dataset/__init__.py`
- `engine/data/dataset/_dataset.py`
- `engine/data/dataset/coco_dataset.py`
- `engine/data/dataset/coco_eval.py`
- `engine/data/dataset/coco_utils.py`
- `engine/data/dataset/voc_detection.py`
- `engine/data/dataset/voc_eval.py`
- `engine/data/transforms/__init__.py`
- `engine/data/transforms/_transforms.py`
- `engine/data/transforms/container.py`
- `engine/data/transforms/functional.py`
- `engine/data/transforms/mosaic.py`
- `engine/deim/__init__.py`
- `engine/deim/box_ops.py`
- `engine/deim/deim.py`
- `engine/deim/deim_criterion.py`
- `engine/deim/deim_decoder.py`
- `engine/deim/deim_utils.py`
- `engine/deim/denoising.py`
- `engine/deim/dfine_decoder.py`
- `engine/deim/dfine_utils.py`
- `engine/deim/hybrid_encoder.py`
- `engine/deim/irdino_modules.py`
- `engine/deim/lite_encoder.py`
- `engine/deim/matcher.py`
- `engine/deim/postprocessor.py`
- `engine/deim/rtdetrv2_decoder.py`
- `engine/deim/utils.py`
- `engine/misc/__init__.py`
- `engine/misc/box_ops.py`
- `engine/misc/dist_utils.py`
- `engine/misc/lazy_loader.py`
- `engine/misc/logger.py`
- `engine/misc/profiler_utils.py`
- `engine/misc/visualizer.py`
- `engine/optim/__init__.py`
- `engine/optim/amp.py`
- `engine/optim/ema.py`
- `engine/optim/lr_scheduler.py`
- `engine/optim/optim.py`
- `engine/optim/warmup.py`
- `engine/solver/__init__.py`
- `engine/solver/_solver.py`
- `engine/solver/clas_engine.py`
- `engine/solver/clas_solver.py`
- `engine/solver/det_engine.py`
- `engine/solver/det_solver.py`
- `eval.py`
- `requirements.txt`
- `scripts/train_daub.sh`
- `scripts/train_irdst.sh`
- `scripts/train_itsdt15k.sh`
- `tools/check_coco_dataset.py`
- `tools/eval_prf1.py`
- `tools/launch_irdino_train.py`
- `train.py`

## Files Excluded

- `datasets/`
- `data/`
- `weights/`
- `checkpoints/`
- `logs/`
- `wandb/`
- `runs/`
- `outputs/`
- `experiments/`
- `results/`
- `*.pth`
- `*.pt`
- `*.ckpt`
- `*.safetensors`

## Known Limitations

- Datasets are not included.
- Pretrained weights and trained checkpoints are not included.
- Real training/evaluation requires users to prepare datasets and weights locally.
- The open-source package preserves the original detector code structure to keep the training path runnable.
