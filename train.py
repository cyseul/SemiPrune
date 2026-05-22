import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
torch.backends.cudnn.benchmark = True
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse
import pickle
import numpy as np
import random
from datetime import datetime
import json
from pathlib import Path
import wandb

from core.model_generator import resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import (
    CoresetSelection,
    IndexDataset,
    CIFARDataset,
    SVHNDataset,
    CINIC10Dataset,
    STL10Dataset,
    CUB200Dataset,
    Food101Dataset,
    Caltech101Dataset,
    SUN397Dataset,
    TinyImageNetDataset,
)
from core.utils import print_training_info, StdRedirect

model_names = ['resnet18', 'wrn-34-10', 'preact_resnet18']
DATASET_CHOICES = [
    'caltech101', 'food101', 'sun397', 'cub200',
    'cifar10', 'cifar100', 'cifar10-C', 'cifar100-C', 'cifar100-LT',
    'tiny-imagenet', 'svhn', 'cinic10', 'stl10',
]
IMAGENET_STYLE_DATASETS = {
    'cub200', 'food101', 'caltech101', 'sun397', 'tiny-imagenet',
}
NUM_CLASSES = {
    'cifar10': 10,
    'cifar10-C': 10,
    'svhn': 10,
    'cinic10': 10,
    'stl10': 10,
    'cifar100': 100,
    'cifar100-C': 100,
    'cifar100-LT': 100,
    'cub200': 200,
    'tiny-imagenet': 200,
    'food101': 101,
    'caltech101': 101,
    'sun397': 397,
}
DEFAULT_DUAL_HARD_SCORE_PATH = 'data-model-0504/sun397/data-score-gt-full-train.pickle'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10,CIFAR100 Training')

######################### Training Setting #########################
parser.add_argument('--epochs', type=int, metavar='N', help='The number of epochs to train a model.')
parser.add_argument('--iterations', type=int, metavar='N', help='The number of iteration to train a model; conflict with --epoch.')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_policy', '--lr-policy', '--scheduler', dest='lr_policy', type=str, default='cosine',
                    choices=['cosine'], help='Learning-rate scheduler policy. Only cosine is supported in this trainer.')
parser.add_argument('--network', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
parser.add_argument('--dataset', type=str, default='cifar10', choices=DATASET_CHOICES)

######################### Print Setting #########################
parser.add_argument('--iterations-per-testing', type=int, default=800, metavar='N', help='The number of iterations for testing model')
parser.add_argument('--ignore-td', action='store_true', default=False)

######################### Path Setting #########################
parser.add_argument('--data-dir', type=str, default='../data/', help='The dir path of the data.')
parser.add_argument('--base-dir', type=str, help='The base dir of this project.')
parser.add_argument('--task-name', type=str, default='tmp', help='The name of the training task.')
parser.add_argument('--caltech-split-path', type=str, default=None, help='Optional split file containing train_indices/test_indices for caltech101')
parser.add_argument('--caltech-split-seed', type=int, default=0, help='Seed used to resolve default split path for caltech101')
parser.add_argument('--cifar100-c-path', type=str,
                    default='data/cifar100-C/img.bin', help='Path to the CIFAR100-C corrupted image pickle')
parser.add_argument('--cifar10-c-path', type=str, default=None, help='Path to the CIFAR10-C corrupted image pickle')
parser.add_argument('--lt-if', type=str, default=None, help='CIFAR100-LT imbalance factor: 0.1 or 0.01')
parser.add_argument('--lt-seed', type=int, default=0, help='Seed used to generate the CIFAR100-LT mask')

######################### Coreset Setting #########################
parser.add_argument('--coreset', action='store_true', default=False)
parser.add_argument('--coreset-mode', type=str, choices=['random', 'coreset', 'stratified', 'swav', 'badge', 'budget', 'hard', 'dual_hard', 'dual_beta', 'abl'])
parser.add_argument('--data-score-path', type=str)
parser.add_argument('--coreset-key', type=str)
parser.add_argument('--data-score-descending', type=int, default=0, help='Set 1 to use larger score data first.')
parser.add_argument('--class-balanced', type=int, default=0, help='Set 1 to use the same class ratio as to the whole dataset.')
parser.add_argument('--coreset-ratio', type=float)
parser.add_argument('--T', '--pseudo-dual-T', dest='T', type=int, help='T value used to select dual_T{T} for dual_hard coreset mode.')
parser.add_argument('--pseudo-dual-p', type=float, default=1.0, help='p value used to select pseudo dual scores for dual_beta coreset mode.')
parser.add_argument('--beta-cd', type=float, default=3.0, help='c_d value used for dual_beta coreset sampling.')

#### Double-end Pruning Setting ####
parser.add_argument('--mis-key', type=str)
parser.add_argument('--mis-data-score-descending', type=bool, default=0, help='Set 1 to use larger score data first.')
parser.add_argument('--mis-ratio', type=float)

#### Reversed Sampling Setting ####
parser.add_argument('--reversed-ratio', type=float, help="Ratio for the coreset, not the whole dataset.")

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0', help='The ID of GPU.')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--wandb-project', type=str, default=None,
                    help='WandB project name. Defaults to <dataset>_<network>.')

################### Load Pseudo Labels from DL models ###################
parser.add_argument('--load-pseudo', action='store_true', default=False)
parser.add_argument('--pseudo-train-label-path', type=str, help='Path for the pseudo train labels')
parser.add_argument('--pseudo-test-label-path', type=str, help='Path for the pseudo test')

######################### Save Coreset Index for Plotting #########################
parser.add_argument('--save-coreset', action='store_true', default=True)
parser.add_argument('--end-early', action='store_true', default=False)

######################### Setting for Future Use #########################
parser.add_argument('--load-from-best', action='store_true', default=False)
# parser.add_argument('--ckpt-name', type=str, default='model.ckpt', help='The name of the checkpoint.')
# parser.add_argument('--lr-scheduler', choices=['step', 'cosine'])
# parser.add_argument('--network', choices=model_names, default='resnet18')
# parser.add_argument('--pretrained', action='store_true')
# parser.add_argument('--augment', choices=['cifar10', 'rand'], default='cifar10')


args = parser.parse_args()
if args.dataset == 'cifar10-C' and args.cifar10_c_path is None:
    parser.error("--cifar10-c-path must be specified when --dataset is cifar10-C")
if args.dataset == 'cifar100-LT' and args.lt_if is None:
    parser.error("--lt-if must be specified when --dataset is cifar100-LT")
start_time = datetime.now()

def resolve_caltech_split_path(data_dir, split_path=None, split_seed=0):
    if split_path:
        return split_path
    root = Path(data_dir).expanduser()
    if root.name.lower() == "caltech101":
        root = root.parent
    return str(root / "caltech101" / f"caltech101_split_seed{int(split_seed)}.pt")


def load_train_dataset(args, data_dir, caltech_split_path=None):
    valset = None

    if args.dataset == 'cifar10':
        trainset = CIFARDataset.get_cifar10_train(data_dir)
    elif args.dataset == 'cifar10-C':
        trainset = CIFARDataset.get_cifar10_c_train(data_dir, args.cifar10_c_path)
    elif args.dataset == 'cifar100':
        trainset = CIFARDataset.get_cifar100_train(data_dir)
    elif args.dataset == 'cifar100-C':
        trainset = CIFARDataset.get_cifar100_c_train(data_dir, args.cifar100_c_path)
    elif args.dataset == 'cifar100-LT':
        trainset = CIFARDataset.get_cifar100_lt_train(data_dir, lt_if=args.lt_if, lt_seed=args.lt_seed)
    elif args.dataset == 'svhn':
        trainset = SVHNDataset.get_svhn_train(data_dir)
    elif args.dataset == 'stl10':
        trainset = STL10Dataset.get_stl10_train(data_dir)
    elif args.dataset == 'cinic10':
        trainset = CINIC10Dataset.get_cinic10_train(data_dir)
        valset = CINIC10Dataset.get_cinic10_train(data_dir, is_val=True)
    elif args.dataset == 'cub200':
        trainset = CUB200Dataset.get_cub200_train(data_dir)
    elif args.dataset == 'food101':
        trainset = Food101Dataset.get_food101_train(data_dir)
    elif args.dataset == 'caltech101':
        trainset = Caltech101Dataset.get_caltech101_train(data_dir, split_path=caltech_split_path, split_seed=args.caltech_split_seed)
    elif args.dataset == 'sun397':
        trainset = SUN397Dataset.get_sun397_train(data_dir)
    elif args.dataset == 'tiny-imagenet':
        trainset = TinyImageNetDataset.get_TinyImageNet_train(data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return trainset, valset


def load_test_dataset(args, data_dir, caltech_split_path=None):
    if args.dataset in ['cifar10', 'cifar10-C']:
        return CIFARDataset.get_cifar10_test(data_dir)
    if args.dataset in ['cifar100', 'cifar100-C', 'cifar100-LT']:
        return CIFARDataset.get_cifar100_test(data_dir)
    if args.dataset == 'svhn':
        return SVHNDataset.get_svhn_test(data_dir)
    if args.dataset == 'stl10':
        return STL10Dataset.get_stl10_test(data_dir)
    if args.dataset == 'cinic10':
        return CINIC10Dataset.get_cinic10_test(data_dir)
    if args.dataset == 'cub200':
        return CUB200Dataset.get_cub200_test(data_dir)
    if args.dataset == 'food101':
        return Food101Dataset.get_food101_test(data_dir)
    if args.dataset == 'caltech101':
        return Caltech101Dataset.get_caltech101_test(data_dir, split_path=caltech_split_path, split_seed=args.caltech_split_seed)
    if args.dataset == 'sun397':
        return SUN397Dataset.get_sun397_test(data_dir)
    if args.dataset == 'tiny-imagenet':
        return TinyImageNetDataset.get_TinyImageNet_test(data_dir)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


PSEUDO_LABEL_LOADERS = {
    'cifar10': CIFARDataset.load_custom_labels,
    'cifar10-C': CIFARDataset.load_custom_labels,
    'cifar100': CIFARDataset.load_custom_labels,
    'cifar100-C': CIFARDataset.load_custom_labels,
    'cifar100-LT': CIFARDataset.load_custom_labels,
    'svhn': SVHNDataset.load_custom_labels,
    'stl10': STL10Dataset.load_custom_labels,
    'cub200': CUB200Dataset.load_custom_labels,
    'food101': Food101Dataset.load_custom_labels,
    'caltech101': Caltech101Dataset.load_custom_labels,
    'sun397': SUN397Dataset.load_custom_labels,
    'tiny-imagenet': TinyImageNetDataset.load_custom_labels,
}


def apply_train_pseudo_labels(args, trainset, valset=None):
    if not args.load_pseudo:
        return trainset, valset
    if not args.pseudo_train_label_path:
        raise ValueError("--pseudo-train-label-path is required when --load-pseudo is set")

    print(f"Loading Pseudo dataset labels from {args.pseudo_train_label_path}")
    if args.dataset == 'cinic10':
        trainset = CINIC10Dataset.load_custom_labels(trainset, args.pseudo_train_label_path)
        if valset is not None:
            valset = CINIC10Dataset.load_custom_labels(valset, args.pseudo_train_label_path, is_val=True)
        return trainset, valset

    trainset = PSEUDO_LABEL_LOADERS[args.dataset](trainset, args.pseudo_train_label_path)
    return trainset, valset


def apply_test_pseudo_labels(args, testset):
    if not args.load_pseudo or not args.pseudo_test_label_path:
        return testset

    print(f"Loading Pseudo dataset labels from {args.pseudo_test_label_path}")
    if args.dataset == 'cinic10':
        return CINIC10Dataset.load_custom_labels(testset, args.pseudo_test_label_path, is_test=True)
    return PSEUDO_LABEL_LOADERS[args.dataset](testset, args.pseudo_test_label_path)

# For reproduction
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

assert args.epochs is None or args.iterations is None, "Both epochs and iterations are used!"

if args.coreset:
    task_name = f"{args.coreset_mode}_sub{args.coreset_ratio}"
    if args.mis_ratio:
        task_name = f"{args.coreset_mode}_sub{args.coreset_ratio}_mis{args.mis_ratio}"
    
wandb.init(
    project=args.wandb_project or f"{args.dataset}_{args.network}",
    name=args.task_name,
    config=vars(args),
)


print(f'Dataset: {args.dataset}')

######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
os.makedirs(task_dir, exist_ok=True)
last_ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
best_ckpt_path = os.path.join(task_dir, f'ckpt-best.pt')
trainset_label_path = os.path.join(task_dir, f'trainset-labels.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
log_path = os.path.join(task_dir, f'log-train-{args.task_name}.log')

######################### Print setting #########################
sys.stdout=StdRedirect(log_path)
print_training_info(args, all=True)
#########################
print(f'Last ckpt path: {last_ckpt_path}')

if torch.cuda.is_available():
    gpu_id = str(args.gpuid).split(',')[0]
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
else:
    device = torch.device('cpu')

data_dir = str(Path(args.data_dir).expanduser())
print(f'Data dir: {data_dir}')

caltech_split_path = None
if args.dataset == 'caltech101':
    caltech_split_path = resolve_caltech_split_path(
        data_dir,
        split_path=args.caltech_split_path,
        split_seed=args.caltech_split_seed,
    )
    print(f'Caltech split path: {caltech_split_path}')

trainset, valset = load_train_dataset(args, data_dir, caltech_split_path=caltech_split_path)
trainset, valset = apply_train_pseudo_labels(args, trainset, valset)

if valset is not None:
    # merge trainset and valset
    trainset = torch.utils.data.ConcatDataset([trainset, valset])


######################### Coreset Selection #########################
coreset_key = args.coreset_key
coreset_ratio = args.coreset_ratio
coreset_descending = (args.data_score_descending == 1)
total_num = len(trainset)


if args.coreset:
    if args.coreset_mode not in ['random', 'swav', 'badge', 'hard', 'dual_hard', 'abl']:
        print(args.coreset_mode)
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)

    if args.coreset_mode == 'random':
        coreset_index = CoresetSelection.random_selection(total_num=len(trainset), num=args.coreset_ratio * len(trainset))

    if args.coreset_mode == 'coreset':
        coreset_index = CoresetSelection.score_monotonic_selection(data_score=data_score, key=args.coreset_key, ratio=args.coreset_ratio, descending=(args.data_score_descending == 1), class_balanced=(args.class_balanced == 1))

    if args.coreset_mode == 'stratified':
        mis_num = int(args.mis_ratio * total_num)
        data_score, score_index = CoresetSelection.mislabel_mask(data_score, mis_key='accumulated_margin', mis_num=mis_num, mis_descending=False, coreset_key=args.coreset_key)

        coreset_num = int(args.coreset_ratio * total_num)
        coreset_index, _ = CoresetSelection.stratified_sampling(data_score=data_score, coreset_key=args.coreset_key, coreset_num=coreset_num)
        coreset_index = score_index[coreset_index]
        print(f'Length of coreset: {len(coreset_index)}')

    if args.coreset_mode == 'budget':
        mis_num = int(args.mis_ratio * total_num)
        coreset_num = int(args.coreset_ratio * total_num)
        high_aum_chop_num = total_num - mis_num - coreset_num
        coreset_index = CoresetSelection.direct_selection(data_score, 
                                                          mis_key=args.mis_key, 
                                                          mis_num=mis_num, 
                                                          mis_descending=args.mis_data_score_descending, 
                                                          coreset_key=args.coreset_key,
                                                          chop_num=high_aum_chop_num)

        print(f'Length of coreset: {len(coreset_index)}')

    if args.coreset_mode == 'hard':
        coreset_num = int(args.coreset_ratio * total_num)
        data_score = np.load(args.data_score_path)
        if len(data_score) != total_num:
            raise ValueError(
                f"Score array length ({len(data_score)}) does not match dataset length ({total_num})."
            )
        print(f'High Priority: {np.sort(data_score)[-10: ]}')
        print(f'Low Priority: {np.sort(data_score)[:10]}')
        coreset_index = np.argsort(data_score)[-coreset_num:].tolist()

    if args.coreset_mode == 'dual_hard':
        if args.T is None:
            raise ValueError("--T is required when --coreset-mode is dual_hard.")
        coreset_num = int(args.coreset_ratio * total_num)
        dual_key = f'dual_T{args.T}'
        data_score_path = args.data_score_path or DEFAULT_DUAL_HARD_SCORE_PATH
        with open(data_score_path, 'rb') as f:
            data_score = pickle.load(f)
        if dual_key not in data_score:
            available_dual_keys = sorted(k for k in data_score.keys() if str(k).startswith('dual_T'))
            raise KeyError(
                f"Score key '{dual_key}' was not found in {data_score_path}. "
                f"Available dual keys: {available_dual_keys}"
            )
        score = data_score[dual_key]
        if torch.is_tensor(score):
            score = score.detach().cpu().numpy()
        else:
            score = np.asarray(score)
        if len(score) != total_num:
            raise ValueError(
                f"Score array length ({len(score)}) does not match dataset length ({total_num})."
            )
        print(f'Using dual hard score path: {data_score_path}')
        print(f'Using dual hard score key: {dual_key}')
        print(f'High Priority: {np.sort(score)[-10: ]}')
        print(f'Low Priority: {np.sort(score)[:10]}')
        coreset_index = np.argsort(score)[-coreset_num:].tolist()

    if args.coreset_mode == 'dual_beta':
        coreset_index = CoresetSelection.beta_sampling(
            data_score=data_score,
            coreset_ratio=args.coreset_ratio,
            p=args.pseudo_dual_p,
            c_d=args.beta_cd,
            max_epoch=args.T,
        )
        coreset_index = np.asarray(coreset_index).astype(int).tolist()
        print(f'Using dual beta score path: {args.data_score_path}')
        print(f'Using dual beta p: {args.pseudo_dual_p}')
        print(f'Using dual beta T: {args.T}')
        print(f'Using dual beta c_d: {args.beta_cd}')
        print(f'Length of coreset: {len(coreset_index)}')

    if args.coreset_mode == 'swav':
        enhance = False
        # load pickle file
        with open(args.data_score_path, 'rb') as f:
            data_score = pickle.load(f)
        # data score: list of [index, distance, pseudo_label assigned by kmeans]
        # sort by distance: descending
        data_score = sorted(data_score, key=lambda x: x[1], reverse=True)
        print(f"Loaded data score from {args.data_score_path}")
        print(f'Length of data score: {len(data_score)}')
        # calculate number of coreset to select
        coreset_num = int(args.coreset_ratio * total_num)
        # select the first coreset_num indices
        if enhance:
            coreset_index = CoresetSelection.select_balanced_coreset_prototypicality(data_score, coreset_num)
        else:
            coreset_index = [x[0] for x in data_score[:coreset_num]]

    if args.coreset_mode == 'badge':
        with open(args.data_score_path, 'r') as f:
            badge_data = [json.loads(line) for line in f.readlines()]
        badge_data_sorted = sorted(badge_data, key=lambda x: x["round"])
        aggregated_indices = []
        total_num_indices_needed = int(len(trainset) * coreset_ratio)
        for round_data in badge_data_sorted[1:]:
            if len(aggregated_indices) >= total_num_indices_needed:
                break
            aggregated_indices.extend(round_data["indices"])

        if len(aggregated_indices) > total_num_indices_needed:
            aggregated_indices = aggregated_indices[:total_num_indices_needed]
        aggregated_indices = [int(idx) for idx in aggregated_indices]
        coreset_index = aggregated_indices
        print(f'Selected {len(trainset)} examples using badge coreset starting from round 1.')
    
    if args.coreset_mode == 'abl':
        coreset_index = np.load(args.data_score_path)
    if args.save_coreset:
        # save the coreset as .pt file -  set as default True
        coreset_index_path = os.path.join(task_dir, f'coreset_index.pt')
        with open(coreset_index_path, 'wb') as f:
            pickle.dump(coreset_index, f)
        print(f'Saved coreset index to {coreset_index_path}')
        if args.end_early:
            sys.exit(0)
    
    trainset = torch.utils.data.Subset(trainset, coreset_index)

######################### Coreset Selection end #########################
trainset = IndexDataset(trainset)
print(f"length of train set - {len(trainset)}")
print("first 100 labels in trainset:")
print([int(trainset[i][1][1]) for i in range(min(100, len(trainset)))])


testset = load_test_dataset(args, data_dir, caltech_split_path=caltech_split_path)
testset = apply_test_pseudo_labels(args, testset)


print(f"length of test set - {len(testset)}")
print('First 100 test label:')
print([int(testset[i][1]) for i in range(min(100, len(testset)))])

if args.dataset in IMAGENET_STYLE_DATASETS:
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=12, persistent_workers=True,
        prefetch_factor=6)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size * 2, shuffle=False, pin_memory=True, num_workers=12, persistent_workers=True,
        prefetch_factor=6)
else:
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)


iterations_per_epoch = len(trainloader)
if args.iterations is None:
    num_of_iterations = iterations_per_epoch * args.epochs
else:
    num_of_iterations = args.iterations

num_classes = NUM_CLASSES[args.dataset]
    
if args.dataset in IMAGENET_STYLE_DATASETS:
    if args.network == 'resnet18':
        print('Using torchvision resnet18.')
        model = torchvision.models.resnet18(pretrained=False, progress=True)
    if args.network == 'resnet34':
        print('Using torchvision resnet34.')
        model = torchvision.models.resnet34(pretrained=False, progress=True)
    if args.network == 'resnet50':
        print('Using torchvision resnet50.')
        model = torchvision.models.resnet50(pretrained=False, progress=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    print(f'Replaced final fc layer with num_classes={num_classes}')
    model = model.to(device)
else:
    if args.network == 'resnet18':
        print('resnet18')
        model = resnet('resnet18', num_classes=num_classes, device=device)
    if args.network == 'resnet34':
        print('resnet34')
        model = resnet('resnet34', num_classes=num_classes, device=device)
    if args.network == 'resnet50':
        print('resnet50')
        model = resnet('resnet50', num_classes=num_classes, device=device)
    model = model.to(device)


criterion = nn.CrossEntropyLoss()

if args.dataset in IMAGENET_STYLE_DATASETS:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_iterations, eta_min=1e-4)
epoch_per_testing = max(args.iterations_per_testing // iterations_per_epoch, 1)

print(f'Total epoch: {num_of_iterations // iterations_per_epoch}')
print(f'Iterations per epoch: {iterations_per_epoch}')
print(f'Total iterations: {num_of_iterations}')
print(f'Epochs per testing: {epoch_per_testing}')

wandb.config.update({
    'task_dir': task_dir,
    'num_classes': num_classes,
    'train_size': len(trainset),
    'test_size': len(testset),
    'iterations_per_epoch': iterations_per_epoch,
    'epoch_per_testing': epoch_per_testing,
}, allow_val_change=True)

trainer = Trainer()
if args.ignore_td:
    TD_logger = None
    print('Ignore training dynamics info.')
else:
    TD_logger = TrainingDynamicsLogger()


current_epoch = 0
best_acc = 0
best_epoch = -1
# check if load from best
if args.load_from_best:
    print('Load from best ckpt')
    state = torch.load(best_ckpt_path)

    model.load_state_dict(state['model_state_dict'])
    current_epoch = state['epoch']
    # report best acc
    test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True)
    best_acc = test_acc
    best_epoch = current_epoch
    print(f'Best acc: {test_acc * 100:.2f}')


while num_of_iterations > 0:
    iterations_epoch = min(num_of_iterations, iterations_per_epoch)
    train_loss, train_acc = trainer.train(
        current_epoch, -1, model, trainloader, optimizer, criterion, scheduler, device,
        TD_logger=TD_logger, log_interval=60, printlog=True
    )

    num_of_iterations -= iterations_per_epoch

    if current_epoch % epoch_per_testing == 0 or num_of_iterations == 0:
        test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True)
        log_payload = {
            'epoch': current_epoch,
            'train/loss': float(train_loss),
            'train/acc': float(train_acc),
            'train/lr': float(optimizer.param_groups[0]['lr']),
            'eval/loss': float(test_loss),
            'eval/acc': float(test_acc),
        }

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)
        log_payload['eval/best_acc'] = float(best_acc)
        log_payload['eval/best_epoch'] = int(best_epoch)
        wandb.log(log_payload, step=current_epoch)
    else:
        wandb.log({
            'epoch': current_epoch,
            'train/loss': float(train_loss),
            'train/acc': float(train_acc),
            'train/lr': float(optimizer.param_groups[0]['lr']),
        }, step=current_epoch)

    current_epoch += 1
    # scheduler.step()

# last ckpt testing
test_loss, test_acc = trainer.test(model, testloader, criterion, device, log_interval=20,  printlog=True)
if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = current_epoch
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch
            }
            torch.save(state, best_ckpt_path)
print('==========================')
print(f'Best acc: {best_acc * 100:.2f}')
print(f'Best acc: {best_acc}')
print(f'Best epoch: {best_epoch}')
print(best_acc)

wandb.summary['final_test_loss'] = float(test_loss)
wandb.summary['final_test_acc'] = float(test_acc)
wandb.summary['best_acc'] = float(best_acc)
wandb.summary['best_epoch'] = int(best_epoch)
wandb.summary['total_time_sec'] = float((datetime.now() - start_time).total_seconds())
######################### Save #########################
state = {
    'model_state_dict': model.state_dict(),
    'epoch': current_epoch - 1
}
torch.save(state, last_ckpt_path)
if not args.ignore_td:
    TD_logger.save_training_dynamics(td_path, data_name=args.dataset)

print(f'Total time consumed: {(datetime.now() - start_time).total_seconds():.2f}')
wandb.finish()
