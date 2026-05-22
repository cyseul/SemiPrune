import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import os
import argparse
import pickle
import glob
import re
from pathlib import Path
from core.utils.misc import calculate_hungarian_misclassification_rate
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser()

######################### Data Setting #########################
parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'food101', 'caltech101', 'tiny-imagenet'])
parser.add_argument('--data-dir', type=str, default='../data/imagenet',
                    help='The dir path of the data.')
parser.add_argument('--base-dir', type=str)
parser.add_argument('--task-name', type=str)
parser.add_argument('--data-score-path', type=str)
parser.add_argument('--caltech-split-path', type=str,
                    default='data/embeddings/Caltech101-dino_vitb16/caltech101_split_seed0.pt',
                    help='Path to Caltech101 train/test split file containing train_indices/test_indices')

################### Load Pseudo Labels from DL models ###################
parser.add_argument('--load-pseudo', action='store_true', default=False)
parser.add_argument('--pseudo-train-label-path', type=str, help='Path for the pseudo train labels')
parser.add_argument('--pseudo-dual-window-size', type=int, default=10,
                    help='Sliding window size for pseudo dual score')
parser.add_argument('--pseudo-dual-p', type=float, nargs='+', default=[1.0],
                    help='One or more exponents for the std term in pseudo dual score')
parser.add_argument('--pseudo-dual-max-epoch', type=int, nargs='+', default=[60],
                    help='One or more TD epoch cutoffs to use when computing pseudo dual')
parser.add_argument('--el2n-max-epoch', type=int, default=10,
                    help='Use TD logs up to this epoch when computing EL2N')

args = parser.parse_args()

def process_td_epoch(td_log, data_importance, epoch_probs, el2n_max_epoch):
    l2_loss = torch.nn.MSELoss(reduction='none')

    for item in td_log:
        epoch = item['epoch']
        output = F.softmax(torch.as_tensor(item['output'], dtype=torch.float32), dim=1)
        index = item['idx'].to(dtype=torch.long)
        label = targets[index]
        batch_idx = torch.arange(output.shape[0])

        target_prob = output[batch_idx, label]
        epoch_probs[index] = target_prob

        predicted = output.argmax(dim=1)
        correctness = (predicted == label).to(dtype=torch.int32)
        data_importance['forgetting'][index] += torch.logical_and(
            data_importance['last_correctness'][index] == 1,
            correctness == 0,
        ).to(dtype=torch.int32)
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += correctness

        masked_output = output.clone()
        masked_output[batch_idx, label] = 0
        other_highest_prob = torch.max(masked_output, dim=1).values
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

        top2 = torch.topk(masked_output, k=2, dim=1).values
        pos_margin = top2[:, 0] - top2[:, 1]
        data_importance['accumulated_margin_pos'][index] += pos_margin

        if epoch <= el2n_max_epoch:
            label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)
            el2n_score = torch.sqrt(l2_loss(label_onehot, output).sum(dim=1))
            data_importance['el2n'][index] += el2n_score

def pseudo_dual(preds, window_size=10, p=1.0):
    preds = torch.as_tensor(preds, dtype=torch.float32)

    if preds.dim() != 2:
        raise ValueError(f"Expected preds to have shape (T, N), got {tuple(preds.shape)}")
    if not (1 <= window_size <= preds.size(0)):
        raise ValueError(f"window_size={window_size} must be in [1, {preds.size(0)}]")

    windows_score = []
    for i in range(preds.size(0) - window_size + 1):
        window = preds[i:i + window_size, :]
        win_std = window.std(dim=0, unbiased=False) ** p
        win_mean = window.mean(dim=0)
        windows_score.append(win_std * (1.0 - win_mean))

    score = torch.stack(windows_score, dim=0).mean(dim=0)
    score_np = score.cpu().numpy().astype(np.float32)
    mask = np.argsort(score_np, kind='stable')
    return score_np, mask

def format_p_value(p):
    return f'{float(p):g}'

def format_t_value(t):
    return str(int(t))

def get_dataset_labels(dataset):
    if hasattr(dataset, 'samples'):
        return [label for _, label in dataset.samples]
    if hasattr(dataset, 'targets'):
        return list(dataset.targets)
    if hasattr(dataset, '_labels'):
        return list(dataset._labels)
    raise AttributeError(f'Cannot infer labels from dataset type: {type(dataset)}')

class PseudoLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, pseudo_labels=None):
        self.dataset = dataset
        original_labels = get_dataset_labels(dataset)
        if pseudo_labels is not None:
            if len(dataset) != len(pseudo_labels):
                raise ValueError(f"The dataset has {len(dataset)} entries but there are {len(pseudo_labels)} pseudo labels.")
            mis_cls = calculate_hungarian_misclassification_rate(pseudo_labels, original_labels)
            print(f"Misclassification rate: {mis_cls:.4f}")
            self.labels = pseudo_labels
        else:
            self.labels = original_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, int(self.labels[idx])


class PathLabelDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = list(image_paths)
        self.targets = [int(x) for x in labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx]

def get_train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def resolve_food101_root(data_dir):
    return data_dir if os.path.basename(os.path.normpath(data_dir)) == 'food101' else os.path.join(data_dir, 'food101')


def resolve_caltech101_root(data_dir):
    data_dir = str(Path(data_dir).expanduser().resolve())
    if os.path.basename(os.path.normpath(data_dir)).lower() == 'caltech101':
        return str(Path(data_dir).parent)
    return data_dir


def build_caltech101_samples(root):
    dset = torchvision.datasets.Caltech101(root=root, target_type="category", download=False)
    image_paths = []
    labels = []
    for i, class_id in enumerate(dset.y):
        category = dset.categories[class_id]
        image_id = dset.index[i]
        image_path = os.path.join(
            dset.root,
            '101_ObjectCategories',
            category,
            f'image_{image_id:04d}.jpg',
        )
        image_paths.append(image_path)
        labels.append(int(class_id))
    return np.asarray(image_paths), np.asarray(labels, dtype=np.int64)

def load_label_file(label_path):
    if label_path.endswith('.npy'):
        return np.load(label_path)
    return torch.load(label_path)

def get_target_tensor(dataset):
    labels = getattr(dataset, 'labels', None)
    if labels is None:
        labels = get_dataset_labels(dataset)
    return torch.as_tensor(labels, dtype=torch.int64)

def build_dataset(dataset_name, data_dir, pseudo_labels=None):
    if dataset_name == 'imagenet':
        dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            get_train_transform(),
        )
    elif dataset_name == 'tiny-imagenet':
        dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            get_train_transform(),
        )
    elif dataset_name == 'food101':
        dataset = torchvision.datasets.Food101(
            root=resolve_food101_root(data_dir),
            split='train',
            download=False,
            transform=get_train_transform(),
        )
    elif dataset_name == 'caltech101':
        root = resolve_caltech101_root(data_dir)
        if not os.path.isfile(args.caltech_split_path):
            raise FileNotFoundError(f'Caltech split file not found: {args.caltech_split_path}')
        split_obj = torch.load(args.caltech_split_path, map_location='cpu')
        if 'train_indices' not in split_obj:
            raise KeyError(f"Expected 'train_indices' in split file: {args.caltech_split_path}")
        image_paths, labels = build_caltech101_samples(root)
        train_indices = np.asarray(split_obj['train_indices'], dtype=np.int64)
        dataset = PathLabelDataset(
            image_paths[train_indices],
            labels[train_indices],
            get_train_transform(),
        )
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    return PseudoLabelDataset(dataset, pseudo_labels=pseudo_labels)

def get_td_paths(base_dir, task_name):
    candidate_dirs = []
    if task_name:
        candidate_dirs.append(os.path.join(base_dir, task_name, 'training-dynamics'))
    candidate_dirs.append(os.path.join(base_dir, 'training-dynamics'))

    td_paths = []
    searched_patterns = []
    for td_dir in candidate_dirs:
        td_glob = os.path.join(td_dir, 'td-*.pickle')
        searched_patterns.append(td_glob)
        td_paths = glob.glob(td_glob)
        if td_paths:
            break

    if not td_paths:
        raise FileNotFoundError(
            'No training dynamics files found. Checked patterns: '
            + ', '.join(searched_patterns)
        )

    def extract_epoch(path):
        match = re.search(r'-(\d+)\.pickle$', path)
        if match is None:
            raise ValueError(f'Cannot parse epoch from path: {path}')
        return int(match.group(1))

    return sorted(td_paths, key=extract_epoch)

#Load all data
data_dir = args.data_dir
train_labels = None
if args.load_pseudo:
    print(f'Loading pseudo labels from {args.pseudo_train_label_path}')
    train_labels = load_label_file(args.pseudo_train_label_path)
trainset = build_dataset(args.dataset, args.data_dir, pseudo_labels=train_labels)

targets = get_target_tensor(trainset)
print('First 100 train label:')
print(targets[:100].tolist())
print(f'Loaded label info directly from dataset: {len(targets)} samples')

data_importance = {}
data_size = targets.shape[0]
num_classes = 1000 if args.dataset == 'imagenet' else 200 if args.dataset == 'tiny-imagenet' else 101

data_importance['targets'] = targets.type(torch.int32)
data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)
data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)
data_importance['accumulated_margin_pos'] = torch.zeros(data_size).type(torch.float32)

target_probs_over_time = []

td_paths = get_td_paths(args.base_dir, args.task_name)

for td_path in td_paths:
    print(td_path)
    with open(td_path, 'rb') as f:
         td_data = pickle.load(f)
    epoch_probs = torch.zeros(data_size, dtype=torch.float32)
    process_td_epoch(
        td_data['training_dynamics'],
        data_importance,
        epoch_probs,
        el2n_max_epoch=args.el2n_max_epoch,
    )
    target_probs_over_time.append(epoch_probs)

target_probs_over_time = torch.stack(target_probs_over_time, dim=0)
data_importance['target_probs'] = target_probs_over_time.numpy().astype(np.float32)

p_values = [float(p) for p in args.pseudo_dual_p]
t_values = [int(t) for t in args.pseudo_dual_max_epoch]

pseudo_dual_scores_by_t = {}
pseudo_dual_masks_by_t = {}
pseudo_dual_target_probs_by_t = {}

for t in t_values:
    pseudo_dual_target_probs = target_probs_over_time[:t]
    if pseudo_dual_target_probs.size(0) == 0:
        raise ValueError(f'No TD logs available for pseudo dual computation at T={t}.')

    t_key = format_t_value(t)
    print(f'Using first {pseudo_dual_target_probs.size(0)} epochs for pseudo dual at T={t_key}')

    pseudo_dual_scores = {}
    pseudo_dual_masks = {}
    for p in p_values:
        p_key = format_p_value(p)
        print(f'Computing pseudo dual score for T={t_key}, p={p_key}')
        pseudo_dual_score, pseudo_dual_mask = pseudo_dual(
            pseudo_dual_target_probs,
            window_size=args.pseudo_dual_window_size,
            p=p,
        )
        pseudo_dual_scores[p_key] = pseudo_dual_score
        pseudo_dual_masks[p_key] = pseudo_dual_mask

    pseudo_dual_scores_by_t[t_key] = pseudo_dual_scores
    pseudo_dual_masks_by_t[t_key] = pseudo_dual_masks
    pseudo_dual_target_probs_by_t[t_key] = pseudo_dual_target_probs.numpy().astype(np.float32)

# Keep the legacy keys for backward compatibility using the first requested T and p.
default_t_key = format_t_value(t_values[0])
default_p_key = format_p_value(p_values[0])
data_importance['pseudo_dual'] = pseudo_dual_scores_by_t[default_t_key][default_p_key]
data_importance['pseudo_dual_mask'] = pseudo_dual_masks_by_t[default_t_key][default_p_key]
data_importance['pseudo_dual_p'] = p_values[0]
data_importance['pseudo_dual_scores'] = pseudo_dual_scores_by_t[default_t_key]
data_importance['pseudo_dual_masks'] = pseudo_dual_masks_by_t[default_t_key]
data_importance['pseudo_dual_p_values'] = np.asarray(p_values, dtype=np.float32)
data_importance['pseudo_dual_target_probs'] = pseudo_dual_target_probs_by_t[default_t_key]
data_importance['pseudo_dual_max_epoch'] = int(default_t_key)
data_importance['pseudo_dual_scores_by_T'] = pseudo_dual_scores_by_t
data_importance['pseudo_dual_masks_by_T'] = pseudo_dual_masks_by_t
data_importance['pseudo_dual_target_probs_by_T'] = pseudo_dual_target_probs_by_t
data_importance['pseudo_dual_T_values'] = np.asarray(t_values, dtype=np.int32)

data_score_path = args.data_score_path
print(f'Saving data score at {data_score_path}')
with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)
