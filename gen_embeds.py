"""
Precomputes embeddings for a given model and dataset.
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.backends import cudnn
from tqdm import tqdm

from eval_cluster_utils import knn_classifier
from loaders import get_dataset
from model_builders import load_model


LT_ROOT_DEFAULT = Path(__file__).resolve().parents[1] / "Semi-supervised-learning" / "saved_models" / "classic_cv"


def normalize_lt_if(lt_if: str | None):
    if lt_if is None:
        return None
    value = str(lt_if).lower()
    mapping = {
        "0.1": ("if01", "if0p1"),
        "0.10": ("if01", "if0p1"),
        "if01": ("if01", "if0p1"),
        "if0.1": ("if01", "if0p1"),
        "0.01": ("if001", "if0p01"),
        "if001": ("if001", "if0p01"),
        "if0.01": ("if001", "if0p01"),
    }
    if value not in mapping:
        raise ValueError(f"Unsupported --lt-if value: {lt_if}")
    return mapping[value]


def resolve_datapath(args):
    return args.datapath


def resolve_lt_mask(args):
    if getattr(args, "lt_mask", None):
        return args.lt_mask
    if args.dataset not in {"CIFAR10-LT", "CIFAR100-LT"}:
        return None

    normalized = normalize_lt_if(getattr(args, "lt_if", None))
    if normalized is None:
        raise ValueError("--lt-mask or --lt-if is required for LT datasets")
    _, if_token = normalized
    base_name = args.dataset.lower()
    root = Path(args.lt_root).expanduser().resolve()
    pattern = f"**/{base_name}_{if_token}_seed{args.lt_seed}_lt_indices.npy"
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not find LT index with pattern {pattern} under {root}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple LT indices matched {pattern}: {matches}")
    return str(matches[0])


def dataset_output_name(args):
    name = args.dataset
    if args.dataset in {"CIFAR10-LT", "CIFAR100-LT"}:
        if getattr(args, "lt_if", None):
            if_tag, _ = normalize_lt_if(args.lt_if)
            name = f"{name}-{if_tag}-seed{args.lt_seed}"
        elif getattr(args, "lt_mask", None):
            name = f"{name}-{Path(args.lt_mask).stem}"
    return name


def dataset_num_classes(dataset_name: str):
    mapping = {
        "CIFAR10": 10,
        "CIFAR10-C": 10,
        "CIFAR10-LT": 10,
        "CIFAR20": 20,
        "CIFAR100": 100,
        "CIFAR100-C": 100,
        "CIFAR100-LT": 100,
        "STL10": 10,
        "SVHN": 10,
        "MNIST": 10,
        "TinyImageNet": 200,
        "SUN397": 397,
        "DTD": 47,
        "EuroSAT": 10,
        "FOOD101": 101,
        "Caltech101": 101,
        "caltech101": 101,
    }
    if dataset_name not in mapping:
        raise KeyError(f"Unknown num_classes mapping for {dataset_name}")
    return mapping[dataset_name]


@torch.no_grad()
def compute_embedding(model, loader):
    embeds = []
    labels = []
    for images, label in tqdm(loader):
        images = images.cuda()
        image_features = model(images).float()
        
        embeds.append(image_features.cpu())
        labels.append(label)
    return torch.cat(embeds), torch.cat(labels)

@torch.no_grad()
def compute_neighbors(embedding, k):
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    num_embeds = embedding.shape[0]
    if num_embeds <= 8*1e4:
        dists = embedding @ embedding.permute(1, 0)
        # exclude self-similarity
        dists.fill_diagonal_(-torch.inf)
        return dists.topk(k, dim=-1)   
    else:
        topk_knn_ids = []
        topk_knn_dists = []
        print("Chunk-wise implementation of k-nn in GPU")
        # num_chunks = 12000 
        step_size = 64 # num_embeds // num_chunks
        embedding = embedding.cuda()
        for idx in tqdm(range(0, num_embeds, step_size)):
            idx_next_chunk = min((idx + step_size), num_embeds)
            features = embedding[idx : idx_next_chunk, :]
            # calculate the dot product dist
            dists_chunk = torch.mm(features, embedding.T).cpu()
            dists_chunk.fill_diagonal_(-torch.inf)
            max_dists, indices = dists_chunk.topk(k, dim=-1)
            topk_knn_ids.append(indices)
            topk_knn_dists.append(max_dists)
        return torch.cat(topk_knn_dists), torch.cat(topk_knn_ids)
    
        
def get_outpath(arch, dataset, datapath='data'):
    datapath = Path(datapath).expanduser().resolve()
    arch = arch.replace('/', '_')
    dataset = dataset.replace('/', '_')
    return datapath / 'embeddings' / f'{dataset}-{arch}'


def get_nn(args, preprocess, model, test=False):
    datapath = resolve_datapath(args)
    if args.dataset in ["CIFAR10-C", "CIFAR100-C"]:
        corrupted_mask = args.corrupted_mask
    else:
        corrupted_mask = None

    if args.dataset in ["CIFAR10-LT", "CIFAR100-LT"]:
        lt_mask = resolve_lt_mask(args)
    else:
        lt_mask = None

    caltech_split_file = getattr(args, "caltech_split_file", None)
    caltech_split_seed = getattr(args, "caltech_split_seed", 0)
        
    # dset = get_dataset(args.dataset, datapath=datapath, train=not test, transform=preprocess, download=True)
    dset = get_dataset(
        args.dataset,
        datapath=datapath,
        train=not test,
        transform=preprocess,
        download=True,
        corrupted_mask=corrupted_mask,
        lt_mask=lt_mask,
        caltech_split_seed=caltech_split_seed,
        caltech_split_file=caltech_split_file,
    )
    
    dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=16)
    embeddings, label = compute_embedding(model, dataloader)    
    embeddings = embeddings.squeeze()
    num_classes = dataset_num_classes(args.dataset)
    k = args.k or len(dset) // num_classes
    
    print(f'Computing {k}-NN')
    nn_dists, neighbors = compute_neighbors(embeddings, k)
    # rewrite for cinic10
    return embeddings, label, nn_dists, neighbors, num_classes


def compute_stats(outpath):
    for test in True, False:
        test_str = '-test' if test else ''
        embeddings = torch.load(outpath / f'embeddings{test_str}.pt', map_location='cpu')
        torch.save(embeddings.mean(dim=0), outpath / f'mean{test_str}.pt')
        torch.save(embeddings.std(dim=0), outpath / f'std{test_str}.pt')

def main(args):
    cudnn.benchmark = True
    cudnn.deterministic = True
    modelname = args.arch

    outpath = get_outpath(modelname, dataset_output_name(args), args.outpath)
    if str(args.dataset).lower() in {"caltech101", "caltech_101"}:
        split_seed = int(getattr(args, "caltech_split_seed", 0))
        print(f"Using Caltech101 stratified 8:2 split (seed={split_seed})")
    if args.stats_only:
        compute_stats(outpath)
        return

    model, preprocess = load_model(args, head=False)
    model = model.cuda()
    model.eval()
    
    outpath.mkdir(parents=True, exist_ok=True)

    embs = {}
    labels = {}
    
    # for test in True, False:
    for test in False, True:
        print('Computing', 'test' if test else 'train', 'dataset embedding')
        embeddings, label, nn_dists, neighbors, num_classes = get_nn(args, preprocess, model, test)
        embeddings, label, nn_dists, neighbors = embeddings.cpu(), label.cpu(), nn_dists.cpu(), neighbors.cpu()

        embs[test] = embeddings
        labels[test] = label
        test_str = '-test' if test else ''
        torch.save(embeddings, outpath / f'embeddings{test_str}.pt')
        torch.save(label, outpath / f'label{test_str}.pt')
        torch.save(neighbors, outpath / f'knn{test_str}.pt')
        torch.save(nn_dists, outpath / f'knn_dists{test_str}.pt')
        torch.save(embeddings.mean(dim=0), outpath / f'mean{test_str}.pt')
        torch.save(embeddings.std(dim=0), outpath / f'std{test_str}.pt')
    
    if not args.no_eval_knn:
        print('Computing KNN accuracy')
        top1, top5 = knn_classifier(
            train_features=embs[False],
            train_labels=labels[False],
            test_features=embs[True],
            test_labels=labels[True],
            k=args.classifier_k,
            T=args.temperature,
            num_classes=num_classes
        )
        print(f'Top-1 accuracy: {top1}, Top-5 accuracy: {top5}')
        with open(outpath / 'accuracy.json', 'w') as f:
            json.dump({'top1': top1, 'top5': top5}, f)
    
    # empty gpu memory
    model = model.cpu()
    del model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR100', 'CIFAR10', "STL10", 'CIFAR100-C', 'CIFAR10-C', 'CIFAR100-LT', 'CIFAR10-LT', \
                                                                "CIFAR20", "IN1K", "IN50", 'IN100', "IN200", "IN1K","CINIC10", "SVHN", "MNIST", \
                                                                "FOOD101", "Caltech101", "caltech101", \
                                                                "TinyImageNet", "SUN397", "DTD", "EuroSAT", ""], type=str)
    parser.add_argument('--arch', default='clip_ViT-B/16')
    parser.add_argument('--outpath', type=Path, default=Path('data'))
    parser.add_argument('--temperature', default=0.02, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--classifier-k', default=20, type=int, help='Numbers of neighbors to use in the classifier')
    parser.add_argument('--k', type=int, default=None, help='total NNs to compute. Default: num images / num classes')
    parser.add_argument('--vit_image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--data-dir', '--datapath', dest='datapath', default='../data', type=str)
    parser.add_argument('--corrupted-mask', default=None, type=str)
    parser.add_argument('--lt-mask', default=None, type=str)
    parser.add_argument('--lt-if', default=None, type=str, help='LT imbalance factor alias: if01/if001 or 0.1/0.01')
    parser.add_argument('--lt-seed', default=0, type=int)
    parser.add_argument('--lt-root', default=str(LT_ROOT_DEFAULT), type=str)
    parser.add_argument('--caltech-split-seed', default=0, type=int,
                        help='Seed for class-wise 8:2 train/test split on Caltech101')
    parser.add_argument('--no_eval_knn', action='store_true', help='Do not evaluate k-nn accuracy', default=False)
    parser.add_argument('--stats_only', action='store_true',
                        help='Only compute the mean and std of the dataset for precomputed embeddings')

    main(parser.parse_args())
