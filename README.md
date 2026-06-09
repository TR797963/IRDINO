# IRDINO

Official implementation of IRDINO for moving infrared small target detection.

IRDINO adapts a real-time detection pipeline for moving infrared small target detection. The open-source package includes the training and evaluation code, IRDINO configuration templates, dataset checking utilities, and documentation needed to reproduce the experiment workflow.

## Modules

- `SFTB`: spatial fine-tuning branch.
- `SMAHE`: second-order motion-aware hybrid encoder.
- `SMAM`: second-order motion-aware module.

## Default Settings

- Clip length: `T=5`
- Input resize: `512x512`
- Single-GPU batch size for N configs: `48`
- OOM fallback batch sizes for N configs: `32`, `24`, `16`, `8`
- Single-GPU batch size for L configs: `16`
- Metrics: AP50, Precision, Recall, F1

## Installation

```bash
conda create -n irdino python=3.10 -y
conda activate irdino
pip install -r requirements.txt
```

## Repository Layout

```text
IRDINO-open/
|-- configs/
|   |-- irdino/
|   |-- irstd/
|   |-- dataset_irstd/
|   |-- base/
|   `-- runtime.yml
|-- engine/
|-- tools/
|-- scripts/
|-- docs/
|-- train.py
|-- eval.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- .gitignore
`-- OPEN_SOURCE_CHECK.md
```

## Dataset Layout

Datasets are not included in this repository. Please prepare the datasets following the directory structure below and update the paths in the config files if needed.

```text
datasets/
|-- ITSDT-15K/
|   |-- annotations/
|   |   |-- instances_train2017.json
|   |   `-- instances_test2017.json
|   |-- train2017/
|   `-- test2017/
|-- IRDST/
|   |-- instances_train2017.json
|   |-- instances_test2017.json
|   |-- train2017/
|   `-- test2017/
`-- DAUB/
    |-- annotations/
    |   |-- instances_train2017.json
    |   `-- instances_test2017.json
    |-- train2017/
    `-- test2017/
```

## Weights

Pretrained weights and trained checkpoints are not included in this repository. Please place downloaded or self-trained weights under `weights/` or specify the checkpoint path in the config.

Expected default paths:

```text
weights/deimv2_hgnetv2_n_coco.pth
weights/deimv2_dinov3_l_coco.pth
```

## Training

```bash
python train.py -c configs/irdino/irdino_smahe_n_itsdt15k.yml
python train.py -c configs/irdino/irdino_smahe_n_irdst.yml
python train.py -c configs/irdino/irdino_smahe_n_daub.yml
```

L-model examples:

```bash
python train.py -c configs/irdino/irdino_smahe_l_itsdt15k.yml
python train.py -c configs/irdino/irdino_smahe_l_irdst.yml
python train.py -c configs/irdino/irdino_smahe_l_daub.yml
```

Use the launcher for automatic OOM batch-size fallback:

```bash
python tools/launch_irdino_train.py -c configs/irdino/irdino_smahe_n_itsdt15k.yml --device cuda
```

## Evaluation

```bash
python eval.py -c configs/irdino/irdino_smahe_n_itsdt15k.yml -r path/to/checkpoint.pth
python eval.py -c configs/irdino/irdino_smahe_n_irdst.yml -r path/to/checkpoint.pth
python eval.py -c configs/irdino/irdino_smahe_n_daub.yml -r path/to/checkpoint.pth
```

## Metrics

We report AP50, Precision, Recall, and F1. Precision/Recall/F1 are computed with IoU=0.5 using one-to-one matching between predictions and ground truths.

Important config keys:

- `T: 5`
- `input_size: 512`
- `batch_size`
- `use_sftb`
- `use_smahe`
- `use_smam`
- `conf_threshold`
- `iou_threshold_for_prf1`

## Citation

```bibtex
@article{irdino2026,
  title={IRDINO: Adapting DINOv3 with Second-Order Motion Awareness for Moving Infrared Small Target Detection},
  author={...},
  journal={...},
  year={2026}
}
```

## License

This project follows the license included in `LICENSE`. Third-party components retain their original notices where present in source files.
