# SemiPrune

<<<<<<< HEAD
This repository provides an implementation of SemiPrune, a label-efficient dataset pruning framework that enables existing supervised pruning methods to be applied without full annotation.

SemiPrune first uses a small randomly labeled subset to train a semi-supervised learning model and generate pseudo-labels for unlabeled data. It then applies supervised pruning methods to the resulting pseudo-labeled training pool.
=======
Research code for SSL-based pseudo-labeling, importance-score generation, and coreset training experiments on CIFAR, TinyImageNet, ImageNet, and related vision datasets.

## Overview

This repository contains two connected experiment pipelines:

1. Our experiments first run SSL to obtain pseudo-labels or SSL-trained representations.
2. ELFS generates pseudo-labels with an embedding + clustering-head pipeline, then uses those pseudo-labels to collect proxy training dynamics and select a coreset.


## Environment
Create the conda environment from the provided file:

```bash
conda env create -f requirements.yml
conda activate elfs
```

The environment is CUDA-oriented and includes PyTorch, torchvision, faiss, scikit-learn, pandas, matplotlib, and related experiment utilities.

Some scripts import `augs.augs`; make sure that module is available in this project or on `PYTHONPATH` before running clustering-head training.

## Repository Layout

```text
.
├── train.py                         # supervised classifier training with optional coreset selection
├── train_imagenet.py                # ImageNet-style supervised training variant
├── generate_importance_score.py     # importance-score generation from training dynamics or embeddings
├── generate_importance_score_imagenet.py
├── gen_embeds.py                    # precompute embeddings, labels, KNN, mean/std
├── train_cluster_heads.py           # teacher-student clustering-head training
├── eval_cluster.py                  # evaluate cluster checkpoints and save pseudo-labels
├── baseline_kmeans.py               # K-Means baseline on embeddings
├── linear_evaluation.py             # linear/MLP evaluation on frozen backbones
├── corrupt-cifar.py                 # generate corrupted CIFAR-100 training data
├── corrupt-tiny.py                  # generate corrupted TinyImageNet-style data
├── core/                            # datasets, model generators, training utilities
├── loaders/                         # embedding/image dataset loaders
├── losses/                          # clustering/self-training losses
└── model_builders/                  # backbone and multi-head model builders
```

## Data

Default dataset roots vary by script:

- Supervised training scripts generally use `--data-dir ../data/`.
- Generated embeddings are saved under `data/embeddings/<dataset>-<arch>/` by default.
- Training outputs are saved under the user-provided `--base-dir/--task-name` or `--output_dir`.

Common supported datasets include CIFAR10, CIFAR100, CIFAR10-C, CIFAR100-C, CIFAR100-LT, STL10, SVHN, CINIC10, TinyImageNet, Caltech101, Food101, SUN397, DTD, EuroSAT, and ImageNet-1K.

## Our Workflow

For our experiments, run the SSL pipeline first. The SSL outputs should provide the pseudo-labels used for downstream training-dynamics collection and coreset selection.

After SSL pseudo-labels are available, use the supervised training and score-generation scripts in this repository with `--load-pseudo`:

```bash
python train.py \
  --dataset cifar10 \
  --gpuid 0 \
  --epochs 200 \
  --lr 0.1 \
  --network resnet18 \
  --batch-size 128 \
  --task-name ssl-pseudo-all-data \
  --base-dir ./data-model/cifar10 \
  --load-pseudo \
  --pseudo-train-label-path <path-to-ssl-pseudo-label.pt> \
  --pseudo-test-label-path <path-to-ssl-pseudo-label-test.pt>
```

Then calculate the importance score:

```bash
python generate_importance_score.py \
  --dataset cifar10 \
  --gpuid 0 \
  --base-dir ./data-model/cifar10 \
  --task-name ssl-pseudo-all-data \
  --load-pseudo \
  --pseudo-train-label-path <path-to-ssl-pseudo-label.pt>
```

Finally, train with the selected coreset:

```bash
python train.py \
  --dataset cifar10 \
  --gpuid 0 \
  --epochs 200 \
  --task-name ssl-budget-0.1 \
  --base-dir ./data-model/cifar10 \
  --coreset \
  --coreset-mode budget \
  --data-score-path ./data-model/cifar10/ssl-pseudo-all-data/data-score-ssl-pseudo-all-data.pickle \
  --coreset-key accumulated_margin \
  --coreset-ratio 0.1 \
  --mis-ratio 0.4 \
  --ignore-td
```

For forgetting-based selection, use `--coreset-key forgetting --data-score-descending 1`.

## ELFS Reference Workflow

ELFS does pseudo-label generation differently: it first generates pseudo-labels without ground-truth labels using pretrained embeddings and clustering heads, then uses those pseudo-labels to collect proxy training dynamics for coreset selection.

### 1. Generate embeddings and nearest neighbors

For example, to use DINO features on CIFAR10:

```bash
python gen_embeds.py \
  --arch dino_vitb16 \
  --dataset CIFAR10 \
  --batch_size 256
```

Generated files are stored under:

```text
data/embeddings/CIFAR10-dino_vitb16/
```

The directory includes train/test embeddings, labels, nearest-neighbor indices, nearest-neighbor distances, and embedding mean/std files.

### 2. Train cluster heads

Train multiple clustering heads using the precomputed embeddings and KNN graph:

```bash
export CUDA_VISIBLE_DEVICES=0
outdir="./experiments/cifar10-dino"
clusters=10
dataset="CIFAR10"

python train_cluster_heads.py \
  --precomputed \
  --arch dino_vitb16 \
  --batch_size 1024 \
  --use_fp16 true \
  --max_momentum_teacher 0.996 \
  --lr 1e-4 \
  --warmup_epochs 20 \
  --min_lr 1e-4 \
  --epochs 200 \
  --output_dir "$outdir" \
  --dataset "$dataset" \
  --knn 50 \
  --out_dim "$clusters" \
  --num_heads 50 \
  --loss TEMI \
  --loss-args beta=0.6 \
  --optimizer adamw
```

### 3. Generate pseudo-labels

```bash
python eval_cluster.py --ckpt_folder "$outdir"
```

The pseudo-labels are saved as:

```text
experiments/cifar10-dino/pseudo_label.pt
experiments/cifar10-dino/pseudo_label-test.pt
```

### 4. Collect training dynamics with pseudo-labels

This step is required before ELFS coreset selection. Load the generated pseudo-labels with `--load-pseudo`.

```bash
python train.py \
  --dataset cifar10 \
  --gpuid 0 \
  --epochs 200 \
  --lr 0.1 \
  --network resnet18 \
  --batch-size 128 \
  --task-name all-data \
  --base-dir ./data-model/cifar10 \
  --load-pseudo \
  --pseudo-train-label-path experiments/cifar10-dino/pseudo_label.pt \
  --pseudo-test-label-path experiments/cifar10-dino/pseudo_label-test.pt
```

This writes checkpoints, logs, and training dynamics to:

```text
./data-model/cifar10/all-data/
```

### 5. Calculate importance scores

Compute ELFS data scores from the pseudo-label training dynamics:

```bash
python generate_importance_score.py \
  --dataset cifar10 \
  --gpuid 0 \
  --base-dir ./data-model/cifar10 \
  --task-name all-data \
  --load-pseudo \
  --pseudo-train-label-path experiments/cifar10-dino/pseudo_label.pt
```

The data score is saved as:

```text
./data-model/cifar10/all-data/data-score-all-data.pickle
```

### 6. Train with ELFS coreset selection

```bash
python train.py \
  --dataset cifar10 \
  --gpuid 0 \
  --epochs 200 \
  --task-name budget-0.1 \
  --base-dir ./data-model/cifar10 \
  --coreset \
  --coreset-mode budget \
  --data-score-path ./data-model/cifar10/all-data/data-score-all-data.pickle \
  --coreset-key accumulated_margin \
  --coreset-ratio 0.1 \
  --mis-ratio 0.4 \
  --ignore-td
```

Useful coreset modes include:

- `random`
- `coreset`
- `stratified`
- `swav`
- `badge`
- `budget`
- `hard`


## Corrupted Datasets

Generate a corrupted CIFAR-100 train split:

```bash
python corrupt-cifar.py \
  --root ../data \
  --save-dir ../data/cifar-100-corrupt \
  --per-corrupt-rate 0.06 \
  --seed 0 \
  --download
```

This creates:

```text
img.bin
targets.bin
```

Use the generated image pickle with scripts that accept `--cifar100-c-path`.

## Notes

- `--base-dir` is required by supervised training and importance-score scripts because task outputs are composed as `<base-dir>/<task-name>`.
- `--epochs` and `--iterations` are mutually exclusive in supervised training.
- For CIFAR100-LT training, pass `--lt-if 0.1` or `--lt-if 0.01`.
- For CIFAR10-C training, pass `--cifar10-c-path`.
- For CIFAR100-C training, pass `--cifar100-c-path` unless the default local path is valid.
- Generated `__pycache__`, checkpoints, embeddings, logs, and experiment outputs should generally not be committed.
>>>>>>> f3b29fb (Initial commit)
