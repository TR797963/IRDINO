# Dataset Preparation

Datasets are not included in this repository.

## Recommended Layout

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

## Annotation Format

IRDINO expects COCO-style JSON annotations with `images`, `annotations`, and `categories` fields. Bounding boxes should use COCO `xywh` format.

## Resolving `file_name`

If `file_name` in the COCO JSON is relative, it is resolved against the dataset image root configured by `img_folder` in the corresponding YAML file.

## Config Paths

Dataset configs live in `configs/dataset_irstd/`. The default open-source paths are:

- `datasets/ITSDT-15K`
- `datasets/IRDST`
- `datasets/DAUB`

Edit `img_folder` and `ann_file` in the YAML files if your datasets are stored elsewhere.

## Custom Datasets

For custom infrared small target datasets, convert annotations to COCO format and create a dataset YAML following the provided templates.
