#!/usr/bin/env bash
set -euo pipefail
python tools/launch_irdino_train.py -c configs/irdino/irdino_smahe_n_irdst.yml --device cuda
