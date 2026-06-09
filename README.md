# IRDINO

IRDINO: Adapting DINOv3 with Second-Order Motion Awareness for Moving Infrared Small Target Detection



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


## License

This project follows the license included in `LICENSE`. Third-party components retain their original notices where present in source files.
