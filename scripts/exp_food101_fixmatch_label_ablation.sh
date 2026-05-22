#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --epochs 200 \
  --dataset food101 \
  --data-dir ../data/food-101 \
  --base-dir ./data-model/food101/fixmatch-ps/lb757 \
  --task-name full-train \
  --gpuid 4 \
  --load-pseudo \
  --pseudo-train-label-path ../Semi-supervised-learning/saved_models/classic_cv/fixmatch_food101_707_0/pseudo_labels.npy &

python train.py \
  --epochs 200 \
  --dataset food101 \
  --data-dir ../data/food-101 \
  --base-dir ./data-model/food101/fixmatch-ps/lb1515 \
  --task-name full-train \
  --gpuid 5 \
  --load-pseudo \
  --pseudo-train-label-path ../Semi-supervised-learning/saved_models/classic_cv/fixmatch_food101_1515_0/pseudo_labels.npy &

python train.py \
  --epochs 200 \
  --dataset food101 \
  --data-dir ../data/food-101 \
  --base-dir ./data-model/food101/fixmatch-ps/lb3738 \
  --task-name full-train \
  --gpuid 6 \
  --load-pseudo \
  --pseudo-train-label-path ../Semi-supervised-learning/saved_models/classic_cv/fixmatch_food101_3838_0/pseudo_labels.npy &

python train.py \
  --epochs 200 \
  --dataset sun397 \
  --data-dir ../data/sun397_full_split/ \
  --base-dir ./data-model/sun397/fixmatch-ps/lb7543 \
  --task-name full-train \
  --gpuid 7 \
  --load-pseudo \
  --pseudo-train-label-path ../Semi-supervised-learning/saved_models/classic_cv/fixmatch_sun397_7543_0_0p80/pseudo_labels.npy &

wait