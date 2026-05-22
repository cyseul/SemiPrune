import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os, sys
import argparse
import pickle
import torch.nn.functional as F
import pandas as pd
from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset, STL10Dataset, CUB_200_2011, Caltech101Dataset, Food101Dataset, SUN397Dataset
from core.utils import print_training_info, find_centroid_kmeans, calculate_distances


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

######################### Data Setting #########################
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training.')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=[
        'cifar10', 'cifar100', 'cifar10-C', 'cifar100-C',
        'cifar100-LT-0.1', 'cifar100-LT-0.01',
        'tiny', 'tiny-imagenet-200-C',
        'svhn', 'cinic10', 'stl10', 'cub200', 'food101', 'caltech101', 'sun397'
    ]
)

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp',
                    help='The name of the training task.')
parser.add_argument('--caltech-split-path', type=str,
                    default='data/embeddings/Caltech101-dino_vitb16/caltech101_split_seed0.pt',
                    help='Path to Caltech101 train/test split file containing train_indices/test_indices')
parser.add_argument('--cifar100-c-path', type=str,
                    default='data/cifar100-C/img.bin',
                    help='Path to the CIFAR100-C corrupted image pickle')
parser.add_argument('--score-batch-size', type=int, default=8192,
                    help='Number of TD examples to process at once on GPU for score computation.')
######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

######################### Importance Score Generation Scheme #########################
parser.add_argument('--from-td', type=int, default=1,
                    help='Set 0 to calculate score for prototypicality.')
parser.add_argument('--importance-scheme', type=str, default='td', choices=['td', 'prototypicality'])  # 
parser.add_argument('--embedding-path', type=str, help='Path for the embedding') # for swav, simclr, etc.

################### Load Pseudo Labels from DL models ###################
parser.add_argument('--load-pseudo', action='store_true', default=False)
parser.add_argument('--pseudo-train-label-path', type=str, help='Path for the pseudo train labels')
parser.add_argument('--td-window-size', type=int, default=10,
                    help='Window size for DynUnc and DUAL.')
parser.add_argument('--dual-T-list', type=str, default='30,60',
                    help='Comma-separated T values for DUAL, e.g., "30,60".')

def parse_int_list(s):
    if s is None or s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

args = parser.parse_args()

######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{args.task_name}.pickle')

######################### Print setting #########################
print_training_info(args, all=True)

#########################
dataset = args.dataset
print(f"Dataset is {dataset}")
if dataset in ['cifar10', 'cifar10-C', 'svhn', 'cinic10', 'stl10']:
    num_classes=10
elif dataset in ['cifar100', 'cifar100-C', 'cifar100-LT-0.1', 'cifar100-LT-0.01']:
    num_classes=100
elif dataset == 'cub200':
    num_classes = 200
elif dataset == 'tiny-imagenet-200-C':
    num_classes = 200
elif dataset == 'food101':
    num_classes = 101
elif dataset == 'caltech101':
    num_classes = 101
elif dataset == 'tiny':
    num_classes = 200
elif dataset == 'sun397':
    num_classes = 397
else:
    raise ValueError(f"Unsupported dataset: {dataset}")
    

######################### Ftn definition #########################
"""Calculate loss and entropy"""
def post_training_metrics(model, dataloader, data_importance, device):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))

    for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        logits = model(inputs)
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()

        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss

"""Calculate td metrics"""
def get_targets_fast(dataset):
    """
    Extract labels without loading images.
    Supports IndexDataset, ConcatDataset, ImageFolder, torchvision datasets,
    and custom datasets used in this file.
    """
    # unwrap IndexDataset-like wrapper
    if hasattr(dataset, "dataset"):
        return get_targets_fast(dataset.dataset)

    # ConcatDataset
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        targets = []
        for d in dataset.datasets:
            targets.extend(get_targets_fast(d).tolist())
        return torch.tensor(targets, dtype=torch.long)

    # Subset
    if isinstance(dataset, torch.utils.data.Subset):
        base_targets = get_targets_fast(dataset.dataset)
        return base_targets[torch.tensor(dataset.indices, dtype=torch.long)]

    # common torchvision datasets
    if hasattr(dataset, "targets"):
        return torch.tensor(np.asarray(dataset.targets, dtype=np.int64), dtype=torch.long)

    if hasattr(dataset, "labels"):
        return torch.tensor(np.asarray(dataset.labels, dtype=np.int64), dtype=torch.long)

    if hasattr(dataset, "_labels"):
        return torch.tensor(np.asarray(dataset._labels, dtype=np.int64), dtype=torch.long)

    # ImageFolder
    if hasattr(dataset, "samples"):
        return torch.tensor([int(y) for _, y in dataset.samples], dtype=torch.long)

    # CUB-style dataframe
    if hasattr(dataset, "data") and isinstance(dataset.data, pd.DataFrame) and "class_id" in dataset.data:
        return torch.tensor(dataset.data["class_id"].astype(np.int64).values, dtype=torch.long)

    raise AttributeError(f"Cannot extract targets without loading images: {type(dataset)}")


def _to_device_tensor(x, device, dtype):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(x, dtype=dtype, device=device)


@torch.no_grad()
def training_dynamics_metrics_gpu(
    td_log,
    dataset,
    data_importance,
    num_classes,
    device,
    el2n_max_epoch=10,
    score_batch_size=8192,
    td_window_size=10,
    dual_T_list=None,
):
    """
    Compute correctness, forgetting, accumulated_margin, and EL2N in one GPU pass.

    This avoids:
    1. loading images to get labels,
    2. computing softmax twice,
    3. doing all score computation on CPU.
    """
    
    targets_cpu = get_targets_fast(dataset)
    data_size = len(targets_cpu)

    target_prob_sum = torch.zeros(data_size, dtype=torch.float32, device=device)
    target_prob_count = torch.zeros(data_size, dtype=torch.int32, device=device)
    target_prob_by_epoch = {}

    targets = targets_cpu.to(device=device, dtype=torch.long, non_blocking=True)
    
    correctness_sum = torch.zeros(data_size, dtype=torch.int32, device=device)
    forgetting = torch.zeros(data_size, dtype=torch.int32, device=device)
    last_correctness = torch.zeros(data_size, dtype=torch.int32, device=device)
    accumulated_margin = torch.zeros(data_size, dtype=torch.float32, device=device)
    el2n = torch.zeros(data_size, dtype=torch.float32, device=device)

    buffer = []
    buffer_n = 0
    current_epoch = None

    def flush_buffer():
        nonlocal buffer, buffer_n

        if len(buffer) == 0:
            return

        outputs = []
        indices = []
        el2n_masks = []

        for item in buffer:
            out = _to_device_tensor(item["output"], device=device, dtype=torch.float32)
            idx = _to_device_tensor(item["idx"], device=device, dtype=torch.long)

            epoch = int(item.get("epoch", -1))
            use_el2n = epoch < el2n_max_epoch

            outputs.append(out)
            indices.append(idx)
            el2n_masks.append(
                torch.full((idx.numel(),), use_el2n, dtype=torch.bool, device=device)
            )

        logits = torch.cat(outputs, dim=0)
        index = torch.cat(indices, dim=0)
        el2n_mask = torch.cat(el2n_masks, dim=0)

        prob = F.softmax(logits, dim=-1)
        label = targets[index]

        pred = prob.argmax(dim=1)
        corr = (pred == label).to(torch.int32)

        # forgetting
        forgetting_now = ((last_correctness[index] == 1) & (corr == 0)).to(torch.int32)
        forgetting[index] += forgetting_now
        last_correctness[index] = corr
        correctness_sum[index] += corr

        # accumulated margin
        target_prob = prob.gather(1, label.view(-1, 1)).squeeze(1)
        target_prob_sum[index] += target_prob
        target_prob_count[index] += 1

        epoch_int = int(current_epoch)
        if epoch_int not in target_prob_by_epoch:
            target_prob_by_epoch[epoch_int] = torch.full(
                (data_size,),
                float("nan"),
                dtype=torch.float32,
                device=device,
            )

        target_prob_by_epoch[epoch_int][index] = target_prob
        prob_for_other = prob.clone()
        prob_for_other.scatter_(1, label.view(-1, 1), -1.0)
        other_highest_prob = prob_for_other.max(dim=1).values

        margin = target_prob - other_highest_prob
        accumulated_margin[index] += margin

        # EL2N without explicit one-hot:
        # ||p - onehot(y)||_2 = sqrt(sum_j p_j^2 - 2p_y + 1)
        if el2n_mask.any():
            idx_el2n = index[el2n_mask]
            prob_el2n = prob[el2n_mask]
            label_el2n = label[el2n_mask]

            target_prob_el2n = prob_el2n.gather(1, label_el2n.view(-1, 1)).squeeze(1)
            el2n_score = torch.sqrt(
                torch.clamp((prob_el2n ** 2).sum(dim=1) - 2.0 * target_prob_el2n + 1.0, min=0.0)
            )
            el2n[idx_el2n] += el2n_score

        buffer = []
        buffer_n = 0

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print(i)

        epoch = int(item.get("epoch", -1))
        n = len(item["idx"])

        # Do not mix different epochs in the same buffer.
        # This keeps forgetting dynamics safe.
        if current_epoch is None:
            current_epoch = epoch
        elif epoch != current_epoch:
            flush_buffer()
            current_epoch = epoch

        buffer.append(item)
        buffer_n += n

        if buffer_n >= score_batch_size:
            flush_buffer()

    flush_buffer()

    # Save CPU tensors for pickle compatibility
    data_importance["target_prob_mean"] = (target_prob_sum / torch.clamp(target_prob_count.float(), min=1.0)).cpu()
    data_importance["targets"] = targets_cpu.to(torch.int32)
    data_importance["correctness"] = correctness_sum.cpu()
    data_importance["forgetting"] = forgetting.cpu()
    data_importance["last_correctness"] = last_correctness.cpu()
    data_importance["accumulated_margin"] = accumulated_margin.cpu()
    data_importance["el2n"] = el2n.cpu()
    
    if len(target_prob_by_epoch) >= td_window_size:
        epochs = sorted(target_prob_by_epoch.keys())
        target_prob_traj = torch.stack(
            [target_prob_by_epoch[e] for e in epochs],
            dim=0,
        )

        # Missing sample values, if any, are set to 0.
        target_prob_traj = torch.nan_to_num(target_prob_traj, nan=0.0)

        # DynUnc: use the full trajectory.
        dynunc_score, dynunc_mask = dynunc(
            target_prob_traj,
            window_size=td_window_size,
            dim=0,
            device=device,
            return_numpy_mask=True,
        )

        data_importance["dynunc"] = dynunc_score
        data_importance["dynunc_mask"] = dynunc_mask

        # DUAL: compute for multiple T values.
        dual_scores, dual_masks = dual_multi_T(
            target_prob_traj,
            T_list=dual_T_list or [],
            window_size=td_window_size,
            dim=0,
            device=device,
            return_numpy_mask=True,
        )

        for T, score_T in dual_scores.items():
            data_importance[f"dual_T{T}"] = score_T
            data_importance[f"dual_T{T}_mask"] = dual_masks[T]
            data_importance[f"target_prob_mean_T{T}"] = target_prob_traj[:T].mean(dim=0).detach().cpu()

    else:
        print(
            f"[WARN] Not enough epochs for DynUnc/DUAL: "
            f"{len(target_prob_by_epoch)} epochs found, "
            f"window_size={td_window_size}."
        )

@torch.no_grad()
def dynunc(preds, window_size=10, dim=0, device=None, return_numpy_mask=True):
    """
    DynUnc over the full trajectory.

    preds shape is usually [num_epochs, num_samples].
    preds[t, i] = target-class probability of sample i at epoch t.
    """
    if device is None:
        device = preds.device if torch.is_tensor(preds) else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    preds = torch.as_tensor(preds, dtype=torch.float32, device=device)

    if dim != 0:
        preds = torch.movedim(preds, dim, 0)

    num_steps = preds.size(0)
    if num_steps < window_size:
        raise ValueError(
            f"window_size={window_size} is larger than number of steps={num_steps}"
        )

    # [num_windows, num_samples, window_size]
    windows = preds.unfold(dimension=0, size=window_size, step=1)

    # Keep original behavior: torch.std default uses unbiased=True.
    win_std = windows.std(dim=-1, unbiased=True) * 10.0

    score = win_std.mean(dim=0)
    mask = torch.argsort(score, descending=False)

    if return_numpy_mask:
        return score.detach().cpu(), mask.detach().cpu().numpy()
    return score, mask


@torch.no_grad()
def dual(preds, window_size=10, dim=0, device=None, return_numpy_mask=True):
    """
    DUAL over the given trajectory.

    If you want DUAL@T, pass preds[:T].
    """
    if device is None:
        device = preds.device if torch.is_tensor(preds) else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    preds = torch.as_tensor(preds, dtype=torch.float32, device=device)

    if dim != 0:
        preds = torch.movedim(preds, dim, 0)

    num_steps = preds.size(0)
    if num_steps < window_size:
        raise ValueError(
            f"window_size={window_size} is larger than number of steps={num_steps}"
        )

    # [num_windows, num_samples, window_size]
    windows = preds.unfold(dimension=0, size=window_size, step=1)

    # Keep original behavior: torch.std default uses unbiased=True.
    win_std = windows.std(dim=-1, unbiased=True) * 10.0
    win_mean = windows.mean(dim=-1)

    score = (win_std * (1.0 - win_mean)).mean(dim=0)
    mask = torch.argsort(score, descending=False)

    if return_numpy_mask:
        return score.detach().cpu(), mask.detach().cpu().numpy()
    return score, mask


@torch.no_grad()
def dual_multi_T(preds, T_list, window_size=10, dim=0, device=None, return_numpy_mask=True):
    """
    Compute DUAL for multiple T values.

    DUAL@T is computed on preds[:T].
    """
    if device is None:
        device = preds.device if torch.is_tensor(preds) else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    preds = torch.as_tensor(preds, dtype=torch.float32, device=device)

    if dim != 0:
        preds = torch.movedim(preds, dim, 0)

    num_steps = preds.size(0)

    dual_scores = {}
    dual_masks = {}

    for T in T_list:
        if T > num_steps:
            print(f"[WARN] Skip DUAL@T={T}: only {num_steps} epochs/checkpoints available.")
            continue

        if T < window_size:
            print(f"[WARN] Skip DUAL@T={T}: T is smaller than window_size={window_size}.")
            continue

        score_T, mask_T = dual(
            preds[:T],
            window_size=window_size,
            dim=0,
            device=device,
            return_numpy_mask=return_numpy_mask,
        )

        dual_scores[T] = score_T
        dual_masks[T] = mask_T

    return dual_scores, dual_masks


#########################

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_identical = transforms.Compose([
            transforms.ToTensor(),
        ])

data_dir =  os.path.join(args.data_dir, dataset)
print(f'dataset: {dataset}, data_dir: {data_dir}')

if args.importance_scheme == 'td':
    valset = None
    if dataset == 'cifar10':
        trainset = CIFARDataset.get_cifar10_train(data_dir, transform=transform_identical)
    elif dataset in ['cifar100', 'cifar100-LT-0.1', 'cifar100-LT-0.01']:
        trainset = CIFARDataset.get_cifar100_train(data_dir, transform=transform_identical)
    elif args.dataset == 'cifar100-C':
        trainset = CIFARDataset.get_cifar100_c_train(
            data_dir,
            args.cifar100_c_path,
            transform=transform_identical,
        )
    elif dataset == 'svhn':
        trainset = SVHNDataset.get_svhn_train(data_dir, transform=transform_identical)
    elif dataset == 'stl10':
        trainset = STL10Dataset.get_stl10_train(data_dir, transform=transform_identical)
    elif args.dataset == 'cinic10':
        trainset = CINIC10Dataset.get_cinic10_train(data_dir, transform=transform_identical)
        valset = CINIC10Dataset.get_cinic10_train(data_dir, transform=transform_identical, is_val=True)
    elif args.dataset == 'cub200':
        trainset = CUB_200_2011(root_dir='../data/CUB_200_2011', train=True, transform=transform_identical)
    elif args.dataset == 'food101':
        trainset = Food101Dataset.get_food101_train(args.data_dir, transform=transform_identical)
    elif args.dataset == 'caltech101':
        trainset = Caltech101Dataset.get_caltech101_train(
            args.data_dir,
            split_path=args.caltech_split_path,
            transform=transform_identical
        )
    elif args.dataset == 'sun397':
        trainset = SUN397Dataset.get_sun397_train(args.data_dir, transform=transform_identical)
    else:
        raise ValueError(f"TD importance is not implemented for dataset: {dataset}")

    if args.load_pseudo:
        if "cifar" in args.dataset:
            #--pseudo_train_label_path example: ../datasets/cifar-100-python/label.pt 
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            trainset = CIFARDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "svhn" in args.dataset:
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            trainset = SVHNDataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "stl" in args.dataset:
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            trainset = STL10Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "food" in args.dataset:
            trainset = Food101Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "cinic" in args.dataset:
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            trainset = CINIC10Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
            print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
            valset = CINIC10Dataset.load_custom_labels(valset, args.pseudo_train_label_path)
        if "cub" in args.dataset:
            trainset = CUB_200_2011.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "caltech" in args.dataset:
            trainset = Caltech101Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if "sun" in args.dataset:
            trainset = SUN397Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)

    if valset:
        # merge trainset and valset
        trainset = torch.utils.data.ConcatDataset([trainset, valset])

    trainset = IndexDataset(trainset)
    print(f"Trainset size: {len(trainset)}")

    data_importance = {}

    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # print(f"Number of classes: {num_classes}")
    # model = resnet('resnet18', num_classes=num_classes, device=device)
    # model = model.to(device)
    
    # print(f'Ckpt path: {ckpt_path}.')
    # checkpoint = torch.load(ckpt_path)['model_state_dict']
    # checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # model.load_state_dict(checkpoint)
    # model.eval()

    with open(td_path, 'rb') as f:
        pickled_data = pickle.load(f)

    training_dynamics = pickled_data['training_dynamics']

    # post_training_metrics(model, trainloader, data_importance, device)
    training_dynamics_metrics_gpu(
        training_dynamics,
        trainset,
        data_importance,
        num_classes=num_classes,
        device=device,
        el2n_max_epoch=20,
        score_batch_size=args.score_batch_size,
        td_window_size=args.td_window_size,
        dual_T_list=parse_int_list(args.dual_T_list),
    )

    print(f'Saving data score at {data_score_path}')
    with open(data_score_path, 'wb') as handle:
        pickle.dump(data_importance, handle)

elif args.importance_scheme == 'prototypicality':
    print("Calculating prototypicality score")
    embeddings = torch.load(args.embedding_path, map_location='cpu')
    print(f"Loading embeddings from {args.embedding_path}, len={len(embeddings)}")
    centroids, labels = find_centroid_kmeans(embeddings, num_classes)
    distances = calculate_distances(embeddings, labels, centroids)

    distances.sort(key=lambda x: x[1], reverse=True)
    # create a data_score_path if it does not exist
    print(f'Saving data score at {data_score_path}, length: {len(distances)}')
    import os 
    os.makedirs(os.path.dirname(data_score_path), exist_ok=True)
    with open(data_score_path, 'wb') as f:
        pickle.dump(distances, f)
else:
    raise ValueError(f"Unsupported importance scheme: {args.importance_scheme}")
