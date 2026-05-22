# SemiPrune

SemiPrune is research code for semi-supervised pseudo-labeling, importance-score generation, and subset training experiments. The typical workflow is:

1. Generate pseudo-labels with an SSL or label-free pipeline.
2. Train a proxy classifier with pseudo-labels and save training dynamics.
3. Generate importance scores from the proxy training dynamics.
4. Train a final model on a selected subset.

## Environment

```bash
conda env create -f requirements.yml
conda activate elfs
```

Some ELFS-style clustering scripts import `augs.augs`; make sure that module is available in this repository or on `PYTHONPATH`.

## Pseudo-Label Sources

For semi-supervised learning, we used [microsoft/Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning.git) to train SSL models and export pseudo-labels. Those pseudo-label files are consumed here with `--load-pseudo`.

ELFS is used as another label-free baseline and is described in the ELFS section below.

## Supported Training Datasets

`train.py` supports:

```text
caltech101, food101, sun397, cub200,
cifar10, cifar100, cifar10-C, cifar100-C, cifar100-LT,
tiny-imagenet, svhn, cinic10, stl10
```

Extra dataset-specific flags:

- `cifar10-C`: pass `--cifar10-c-path`.
- `cifar100-C`: pass `--cifar100-c-path` if the default path is not valid.
- `cifar100-LT`: pass `--lt-if 0.1` or `--lt-if 0.01`.
- `caltech101`: optionally pass `--caltech-split-path`; otherwise the split is resolved from `--data-dir` and `--caltech-split-seed`.

`train_imagenet.py` supports:

```text
imagenet
```

The ImageNet directory should contain `train/` and `val/`.

## Train with Pseudo-Labels

Use `--load-pseudo` to replace the dataset labels with pseudo-labels. The pseudo-labels can come from the Microsoft semi-supervised learning codebase or from the ELFS baseline. For `train.py`, pass both train and test pseudo-labels if you want validation/test to use pseudo-labels too.

```bash
python train.py \
  --dataset cifar100 \
  --data-dir ../data \
  --base-dir ./data-model/cifar100 \
  --task-name pseudo-rn18-200ep-full \
  --gpuid 0 \
  --epochs 200 \
  --lr 0.1 \
  --network resnet18 \
  --batch-size 128 \
  --load-pseudo \
  --pseudo-train-label-path <path-to-pseudo-train-labels.pt> \
  --pseudo-test-label-path <path-to-pseudo-test-labels.pt>
```

For ImageNet:

```bash
python train_imagenet.py \
  --dataset imagenet \
  --data-dir <path-to-imagenet-root> \
  --base-dir ./data-model/imagenet \
  --task-name pseudo-rn34-60ep-full \
  --gpuid 0,1 \
  --epochs 60 \
  --lr 0.1 \
  --scheduler cosine \
  --network resnet34 \
  --batch-size 256 \
  --load-pseudo \
  --pseudo-train-label-path <path-to-imagenet-pseudo-train-labels.pt> \
  --pseudo-test-label-path <path-to-imagenet-pseudo-val-labels.pt>
```

## Generate Importance Scores with Pseudo-Labels

After proxy training, `generate_importance_score.py` reads:

```text
<base-dir>/<task-name>/td-<task-name>.pickle
```

and writes:

```text
<base-dir>/<task-name>/data-score-<task-name>.pickle
```

Example:

```bash
python generate_importance_score.py \
  --dataset cifar100 \
  --data-dir ../data \
  --base-dir ./data-model/cifar100 \
  --task-name pseudo-rn18-200ep-full \
  --gpuid 0 \
  --load-pseudo \
  --pseudo-train-label-path <path-to-pseudo-train-labels.pt> \
  --score-batch-size 8192 \
  --td-window-size 10 \
  --dual-T-list 30,60
```

For ImageNet-style TD logs, use `generate_importance_score_imagenet.py`. This script reads per-epoch TD files under `<base-dir>/<task-name>/training-dynamics/` and requires an explicit `--data-score-path`.

```bash
python generate_importance_score_imagenet.py \
  --dataset imagenet \
  --data-dir <path-to-imagenet-root> \
  --base-dir ./data-model/imagenet \
  --task-name pseudo-rn34-60ep-full \
  --data-score-path ./data-model/imagenet/pseudo-rn34-60ep-full/data-score-pseudo-rn34-60ep-full.pickle \
  --load-pseudo \
  --pseudo-train-label-path <path-to-imagenet-pseudo-train-labels.pt> \
  --pseudo-dual-window-size 10 \
  --pseudo-dual-p 1.0 \
  --pseudo-dual-max-epoch 60
```

## Train on a Selected Subset

`train.py` coreset modes:

- `random`: uniformly samples a subset; does not need `--data-score-path`.
- `coreset`: sorts by `--coreset-key` and takes the top/bottom `--coreset-ratio`, controlled by `--data-score-descending`.
- `stratified`: first removes likely mislabeled examples using `--mis-key`/`--mis-ratio`, then samples across score strata.
- `budget`: SemiPrune AUM-style two-end pruning. It removes `--mis-ratio` low-quality examples by `--mis-key`, chops high-AUM examples outside the budget, and keeps the requested subset.
- `dual_hard`: selects the largest `dual_T<T>` scores from the score pickle.
- `dual_beta`: Semi-DUAL beta sampling. It combines DUAL uncertainty with a beta distribution over target-class probability.
- `swav`: selects by prototypicality/k-means distance score.
- `badge`: consumes BADGE JSONL output.
- `hard`: selects the highest scores from a NumPy score array.
- `abl`: loads a precomputed NumPy index array.

`train_imagenet.py` supports `random`, `coreset`, `stratified`, and `budget`.

### Semi-DUAL-Beta

```bash
python train.py \
  --dataset cifar100 \
  --data-dir ../data \
  --base-dir ./data-model/cifar100 \
  --task-name semi-dual-beta-r10-T60 \
  --gpuid 0 \
  --epochs 200 \
  --lr 0.1 \
  --network resnet18 \
  --batch-size 128 \
  --coreset \
  --coreset-mode dual_beta \
  --data-score-path ./data-model/cifar100/pseudo-rn18-200ep-full/data-score-pseudo-rn18-200ep-full.pickle \
  --coreset-ratio 0.1 \
  --pseudo-dual-T 60 \
  --pseudo-dual-p 1.0 \
  --beta-cd 3.0 \
  --ignore-td
```

### Semi-AUM-Cutoff

```bash
python train.py \
  --dataset cifar100 \
  --data-dir ../data \
  --base-dir ./data-model/cifar100 \
  --task-name semi-aum-cutoff-r10-m40 \
  --gpuid 0 \
  --epochs 200 \
  --lr 0.1 \
  --network resnet18 \
  --batch-size 128 \
  --coreset \
  --coreset-mode budget \
  --data-score-path ./data-model/cifar100/pseudo-rn18-200ep-full/data-score-pseudo-rn18-200ep-full.pickle \
  --mis-key accumulated_margin \
  --mis-data-score-descending 0 \
  --coreset-key accumulated_margin \
  --coreset-ratio 0.1 \
  --mis-ratio 0.4 \
  --ignore-td
```

ImageNet subset example with AUM cutoff:

```bash
python train_imagenet.py \
  --dataset imagenet \
  --data-dir <path-to-imagenet-root> \
  --base-dir ./data-model/imagenet \
  --task-name semi-aum-cutoff-r10-m30 \
  --gpuid 0,1 \
  --iterations 300000 \
  --iterations-per-testing 5000 \
  --lr 0.1 \
  --scheduler cosine \
  --network resnet34 \
  --batch-size 256 \
  --coreset \
  --coreset-mode budget \
  --data-score-path ./data-model/imagenet/pseudo-rn34-60ep-full/data-score-pseudo-rn34-60ep-full.pickle \
  --coreset-key accumulated_margin \
  --coreset-ratio 0.1 \
  --mis-ratio 0.3 \
  --ignore-td
```

## ELFS Baseline Reference

We also reference the ELFS implementation from [eltsai/elfs](https://github.com/eltsai/elfs.git), which provides label-free pseudo-label generation with pretrained embeddings and clustering heads.

The upstream ELFS usage is:

```bash
python gen_embeds.py \
  --arch dino_vitb16 \
  --dataset CIFAR10 \
  --batch_size 256
```

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

```bash
python eval_cluster.py --ckpt_folder "$outdir"
```

This produces pseudo-label files such as:

```text
experiments/cifar10-dino/pseudo_label.pt
experiments/cifar10-dino/pseudo_label-test.pt
```

Then use those files with this repository's pseudo-label training, score generation, and subset-training commands above.

## Notes

- `--base-dir` and `--task-name` define the output directory as `<base-dir>/<task-name>`.
- `--epochs` and `--iterations` are mutually exclusive.
- Use `--ignore-td` for final subset training if you do not need another set of training dynamics.
- Generated checkpoints, TD logs, embeddings, pseudo-labels, and data-score files should generally not be committed.
