import random
import pickle

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.datasets as tds
from pathlib import Path
import os
import loaders
import model_builders
import torchvision
from torchvision import transforms

def _attach_subset_metadata(subset, base_dataset, indices):
    if hasattr(base_dataset, "classes"):
        subset.classes = base_dataset.classes
    if hasattr(base_dataset, "targets"):
        base_targets = np.asarray(base_dataset.targets)
        subset.targets = base_targets[indices].tolist()
    elif hasattr(base_dataset, "labels"):
        base_labels = np.asarray(base_dataset.labels)
        subset.labels = base_labels[indices].tolist()
    return subset


def get_cinic10_dataset(path, transform=None, identity_transform=False, is_val=False, is_test=False):
    if transform is None:
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    if identity_transform:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    if is_test:
        subdir = 'test'
    else:
        subdir = 'valid' if is_val else 'train'
    path = os.path.join(path, subdir)
    return torchvision.datasets.ImageFolder(root=path, transform=transform)

class PrecomputedEmbeddingDataset(Dataset):

    def __init__(self, dataset, arch, train, datapath, embedding_path=None, label_path=None,
                 test_embedding_path=None, test_label_path=None):
        super().__init__()
        if embedding_path:
            if (not train) and test_embedding_path:
                embedding_path = test_embedding_path
            elif not train:
                test_embedding_candidate = Path(embedding_path).with_name("embeddings-test.pt")
                if test_embedding_candidate.is_file():
                    embedding_path = test_embedding_candidate
            self.emb = torch.load(embedding_path, map_location='cpu')
            if not train:
                label_path = test_label_path
            if label_path is None:
                embedding_path = Path(embedding_path)
                suffix = "-test" if not train else ""
                label_candidates = [
                    embedding_path.with_name(f"label{suffix}.pt"),
                    embedding_path.with_name(f"labels{suffix}.pt"),
                ]
                for candidate in label_candidates:
                    if candidate.is_file():
                        label_path = candidate
                        break
            if label_path is None:
                split = "test" if not train else "train"
                expected = " or ".join(str(path) for path in label_candidates)
                raise ValueError(
                    f"Precomputed {split} embeddings require a label file. "
                    f"Expected {expected}, or pass "
                    f"{'--test_label_path' if not train else '--label_path'}."
                )
            if not Path(label_path).is_file():
                raise FileNotFoundError(f"Label file not found: {label_path}")
            self.targets = torch.load(label_path, map_location='cpu')
            if len(self.emb) != len(self.targets):
                split = "test" if not train else "train"
                raise ValueError(
                    f"Embedding/label length mismatch for {split} split: "
                    f"{embedding_path} has {len(self.emb)} rows, but "
                    f"{label_path} has {len(self.targets)} labels."
                )
        else:
            self.emb, self.targets = model_builders.load_embeds(
                arch=arch,
                dataset=dataset,
                datapath=datapath,
                with_label=True,
                test=not train)

    def __getitem__(self, index):
        return self.emb[index], self.targets[index]

    def __len__(self):
        return len(self.emb)


def get_dataset(dataset, datapath='./data', train=True, transform=None, download=True,
                precompute_arch=None, embedding_path=None, label_path=None,
                test_embedding_path=None, test_label_path=None,
                corrupted_mask=None, lt_mask=None):
    if precompute_arch:
        return PrecomputedEmbeddingDataset(
            dataset=dataset,
            arch=precompute_arch,
            datapath="data", # assumes embeddings are saved in the ./data folder
            train=train,
            embedding_path=embedding_path,
            label_path=label_path,
            test_embedding_path=test_embedding_path,
            test_label_path=test_label_path)
    
    load_obj = tds if dataset in ["CIFAR10", "CIFAR100", "CIFAR10-C", "CIFAR100-C", "CIFAR10-LT", "CIFAR100-LT", "STL10", "SVHN", "MNIST"] else loaders
    if dataset == "STL10":
        split = 'train' if train else 'test'
        return getattr(load_obj, dataset)(root=datapath,
                        split=split,
                        download=download, transform=transform)
    elif "CIFAR" in dataset:
        dset = getattr(load_obj, dataset.split('-')[0])(root=datapath,
                        train=train,
                        download=download, transform=transform)
        
        if corrupted_mask is not None:
            if train:
                with open(corrupted_mask, 'rb') as f:
                    corrupted_img = pickle.load(f)
                dset.data = corrupted_img

        if lt_mask is not None:
            if train:
                lt_indices = np.load(lt_mask).astype(np.int64)
                dset = torch.utils.data.Subset(dset, lt_indices)
                dset = _attach_subset_metadata(dset, dset.dataset, lt_indices)
                
        return dset
    
    elif dataset == "SVHN":
        split = 'train' if train else 'test'
        return getattr(load_obj, dataset)(root=datapath, split=split, download=download, transform=transform)
    elif dataset == "MNIST":
        channel_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToPILImage(),  # Convert Tensor to PIL Image before Grayscale
            transforms.ToTensor()
        ])
        # combine transform and channel_transform
        if transform is not None:
            transform = transforms.Compose([channel_transform, transform])
        else:
            transform = channel_transform
        return getattr(load_obj, dataset)(root=datapath, train=train, download=download, transform=transform)
    elif dataset == "CINIC10":
        if train:
            # Combine train and validation sets for training
            train_dataset = getattr(load_obj, dataset)(root=datapath, split='train', download=download, transform=transform)
            val_dataset = getattr(load_obj, dataset)(root=datapath, split='valid', download=download, transform=transform)
            combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
            return combined_dataset
        else:
            return getattr(load_obj, dataset)(root=datapath, split='test', download=download, transform=transform)
    
    else:
        # imagenet subsets
        # TODO i dont know if val and val_structured are the same
        # if "ILSVRC" in datapath and train is False:
            # datapath = datapath.replace("train", "val")
        if train is True:
            datapath = os.path.join(datapath, 'train')
        else:
            datapath = os.path.join(datapath, 'val')
        return getattr(load_obj, dataset)(root=datapath,
                         transform=transform)


class EmbedNN(Dataset):
    def __init__(self,
                 knn_path,
                 transform,
                 k=10,
                 dataset="CIFAR100",
                 datapath='./data',
                 precompute_arch=None,
                 embedding_path=None,
                 label_path=None):
        super().__init__()
        self.transform = transform
        self.complete_neighbors = torch.load(knn_path)
        if k < 0:
            k = self.complete_neighbors.size(1)
        self.k = k
        self.neighbors = self.complete_neighbors[:, :k]
        self.datapath = './data' if 'IN' not in dataset else datapath

        self.dataset = get_dataset(
            dataset,
            datapath=datapath,
            transform=None,
            train=True,
            download=True,
            precompute_arch=precompute_arch,
            embedding_path=embedding_path,
            label_path=label_path)

    def get_transformed_imgs(self, idx, *idcs):
        img, label = self.dataset[idx]
        rest_imgs = (self.dataset[i][0] for i in idcs)
        return self.transform(img, *rest_imgs), label

    def __getitem__(self, idx):
        # KNN pair
        pair_idx = np.random.choice(self.neighbors[idx], 1)[0]

        return self.get_transformed_imgs(idx, pair_idx)

    def __len__(self):
        return len(self.dataset)


class TruePosNN(EmbedNN):

    def __init__(self, knn_path, *args, **kwargs):
        super().__init__(knn_path, *args, **kwargs)
        p = Path(knn_path).parent
        nn_p = p / 'hard_pos_nn.pt'
        if nn_p.is_file():
            self.complete_neighbors = torch.load(nn_p)
        else:
            emb = torch.load(p / 'embeddings.pt')
            emb /= emb.norm(dim=-1, keepdim=True)
            d = emb @ emb.T
            labels = torch.tensor(self.dataset.targets)
            same_label = labels.view(1, -1) == labels.view(-1, 1)
            # Find minimum number of images per class
            k_max = same_label.sum(dim=1).min()
            d.fill_diagonal_(-2)
            d[torch.logical_not(same_label)] = -torch.inf
            self.complete_neighbors = d.topk(k_max, dim=-1)[1]
            torch.save(self.complete_neighbors, nn_p)
        self.neighbors = self.complete_neighbors[:, :self.k]
