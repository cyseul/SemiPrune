import fnmatch
import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path
import os
from typing import Optional, Union

import requests
# import clip
import timm
import torch
import torch.nn as nn
from torchvision import models as torchvision_models, transforms
from tqdm import tqdm

import utils
from .multi_head import MultiHeadClassifier
from timm.models.helpers import load_state_dict


_AVAILABLE_MODELS = (
    "dino_resnet50",
    "dino_vits16",
    "dino_vitb16",
    "timm_resnet50",
    "timm_vit_small_patch16_224",
    "timm_vit_base_patch16_224",
    "timm_vit_large_patch16_224",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "msn_vit_small",
    "msn_vit_base",
    "msn_vit_large",
    "mocov3_vit_small",
    "mocov3_vit_base",
    "clip_ViT-B/16",
    "clip_ViT-L/14",
    "clip_RN50",
    "mae_vit_base",
    "mae_vit_large",
    "mae_vit_huge",
)


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _CifarSwavBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class _CifarSwavResNet50Backbone(nn.Module):
    def __init__(self, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(_CifarSwavBottleneck, 64, 3)
        self.layer2 = self._make_layer(_CifarSwavBottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(_CifarSwavBottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(_CifarSwavBottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=nn.BatchNorm2d,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=nn.BatchNorm2d,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def available_models(pattern=None):
    if pattern is None:
        return _AVAILABLE_MODELS
    return tuple(fnmatch.filter(_AVAILABLE_MODELS, pattern))


def _load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    msg = model.load_state_dict(state_dict, strict=strict)
    print(msg)


def _download(url: str, filename: Path):
    """from https://stackoverflow.com/a/37573701"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception(f"Could not download from {url}")


_dict_models_urls = {
    "msn": {
        'vit_small_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar',
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vitb16_600ep.pth.tar',
        'vit_large_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vitl16_600ep.pth.tar',
        "key": 'target_encoder'
    },
    "mae": {
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        'vit_large_patch16_224': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        'vit_huge_patch14_224_in21k': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        "key": 'model'
    },

    "mocov3": {
        'vit_small_patch16_224': 'https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar',
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar',
        "key":  "state_dict"
    },
}

_dict_timm_names = {
        "vit_huge": 'vit_huge_patch14_224_in21k',
        "vit_large": 'vit_large_patch16_224',
        "vit_base": 'vit_base_patch16_224',
        "vit_small": 'vit_small_patch16_224',
        "vit_tiny": 'vit_tiny_patch16_224',
        "resnet50": 'resnet50',
        }


def _get_checkpoint_path(model_name: str, timm_base=True):
    if timm_base:
        name = _get_timm_name(model_name)
    prefix = model_name.split("_")[0]
    model_url = _dict_models_urls[prefix][name]
    print(f"Loading {model_url}")
    root = Path('~/.cache/torch/checkpoints').expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{model_name}.pth'
    if not path.is_file():
        print('Downloading checkpoint...')
        _download(model_url, path)
        d = torch.load(path, map_location='cpu')
        ckpt_key_name = _dict_models_urls[prefix]["key"]
        if ckpt_key_name in d.keys():
            state_dict = d[ckpt_key_name]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("momentum_encoder.", ""): v for k, v in state_dict.items()} # for mocov3
            torch.save(state_dict, path)
        else:
            raise KeyError(f"{ckpt_key_name} not found. Only {d.keys()} are available.")
    return path


def _get_timm_name(model_name: str):
    prefix = model_name.split("_")[0]
    # remove prefix
    model_name = model_name.replace("".join([prefix,"_"]),"")
    if model_name in _dict_timm_names.keys():
        return _dict_timm_names[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")


def build_arch(model_name: str):
    timm_name = _get_timm_name(model_name)
    model = timm.create_model(
        timm_name,
        in_chans=3,
        num_classes=0,
        pretrained=False)
    _load_checkpoint(model, _get_checkpoint_path(model_name), strict=False)
    return model


def load_model(config, head=True, split_preprocess=False):
    """
    config/args file
    head=False returns just the backbone for baseline evaluation
    split_preprocess=True returns resizing etc. and normalization/ToTensor as separate transforms
    """
    from main_args import set_default_args
    config = set_default_args(config)

    preprocess = None

    if config.precomputed:
        backbone = config.arch
    elif "timm" in config.arch:  # timm models
        arch = config.arch.replace("timm_", "")
        arch = arch.replace("timm-", "")
        backbone = timm.create_model(arch, pretrained=True, in_chans=3, num_classes=0)
    elif "swav" in config.arch:
        if getattr(config, "pretrained_weights", ""):
            # Avoid downloading the default SWaV weights when we already have
            # a local checkpoint to load. CIFAR SWaV checkpoints use a custom
            # ResNet-50 stem (3x3 conv, no maxpool), so match that shape.
            backbone = _CifarSwavResNet50Backbone()
        else:
            backbone = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            backbone.head = None
        
    elif "swag" in config.arch:
        arch = config.arch.replace("swag_", "")
        backbone = torch.hub.load("facebookresearch/swag", model=arch)
        backbone.head = None
    elif "dinov2" in config.arch:
        # dinov2_vitg14, dinov2_vitg14_reg
        print("Loading DINOv2 model")
        arch = config.arch.replace("-", "_")
        backbone = torch.hub.load('facebookresearch/dinov2', arch)
    elif "dino" in config.arch:  # dino pretrained models on IN
        # dino_vitb16, dino_vits16
        arch = config.arch.replace("-", "_")
        backbone = torch.hub.load('facebookresearch/dino:main', arch)
    # elif "clip" in config.arch: # load clip vit models from openai
    #     arch = config.arch.replace("clip_", "")
    #     arch = arch.replace("clip-", "")
    #     assert arch in clip.available_models()
    #     clip_model, preprocess = clip.load(arch)
    #     backbone = clip_model.visual
    
    elif "mae" in config.arch or "msn" in config.arch or "mocov3" in config.arch:
        backbone = build_arch(config.arch)
    elif "convnext" in config.arch:
        backbone = getattr(torchvision_models, config.arch)(pretrained=True)
        backbone.classifier = torch.nn.Flatten(start_dim=1, end_dim=-1)

    elif config.arch in torchvision_models.__dict__.keys(): # torchvision models
        backbone = torchvision_models.__dict__[config.arch](num_classes=0)
    else:
        print(f"Architecture {config.arch} non supported")
        sys.exit(1)
    
    if not config.precomputed:
        print(f"Backbone {config.arch} loaded.")
    else:
        print("No backbone loaded, using precomputed embeddings from", config.arch)

    if preprocess is None:
        # imagenet means/stds
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        resize = transforms.Resize(config.vit_image_size, transforms.InterpolationMode.BICUBIC)
        if config.dataset=="IN1K":
            preprocess = transforms.Compose([
                transforms.Resize(int(256 * config.vit_image_size / 224)),
                transforms.CenterCrop(config.vit_image_size)
            ])
        else:
            preprocess = resize
        if not split_preprocess:
            preprocess = transforms.Compose([
                preprocess,
                transforms.ToTensor(),
                normalize
            ])
    elif split_preprocess:
        preprocess_aux = preprocess.transforms[:-2]
        normalize_aux = preprocess.transforms[-2:]
        preprocess = transforms.Compose(preprocess_aux)
        normalize = transforms.Compose(normalize_aux)

    if head:
        if getattr(config, "embed_dim", None) is None:
            config.embed_dim = utils.embed_dim(config, backbone)
        # Just get everything via reflection
        mmc_params = inspect.signature(MultiHeadClassifier).parameters
        mmc_args = {k: v for k, v in config.__dict__.items() if k in mmc_params}
        model = MultiHeadClassifier(backbone, **mmc_args)

        if config.embed_norm:
            model.set_mean_std(*load_embed_stats(config, test=False))
        print("Head loaded.")
    else:
        model = backbone.float()

    if split_preprocess:
        return model, preprocess, normalize 
    print(preprocess)   
    return model, preprocess


def _build_from_config(
        precomputed: bool,
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    if isinstance(config, str) or isinstance(config, Path):
        p = Path(config)
        with open(p, "r") as f:
            config = json.load(f)
        config = Namespace(**config)
    if config is None:
        config = Namespace()
    config.num_heads = 1
    config.precomputed = precomputed
    if ckpt_path is not None:
        # Don't reload norms
        config.embed_norm = False

    d = None
    if ckpt_path is not None:
        d = torch.load(ckpt_path, map_location="cpu")
        if 'teacher' in d:
            d = d['teacher']
        if 'head.best_head_idx' in d:
            best_head_idx = d['head.best_head_idx']
            d2 = {k: v for k, v in d.items() if k in ('embed_mean', 'embed_std')}
            d2['head.best_head_idx'] = torch.tensor(0)
            for k, v in d.items():
                if k.startswith(f'head.heads.{best_head_idx}.'):
                    k = 'head.heads.0.' + k[len(f'head.heads.{best_head_idx}.'):]
                    d2[k] = v
            d = d2
        else:
            d['head.best_head_idx'] = torch.tensor(0)
        config.embed_dim = d['head.heads.0.mlp.0.weight'].size(1)

    model, _ = load_model(config, head=True)
    model.eval()
    if d is not None:
        model.load_state_dict(d, strict=False)

    return model


def build_head_from_config(
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    """
    config: Either path to hp.json or config namespace
    ckpt_path: Path to checkpoint
    """
    return _build_from_config(True, config, ckpt_path)


def build_model_from_config(
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    """
    config: Either path to hp.json or config namespace
    ckpt_path: Path to checkpoint
    """
    return _build_from_config(False, config, ckpt_path)


def load_embeds(config=None,
                arch=None,
                dataset=None,
                test=False,
                norm=False,
                datapath='data',
                with_label=False):
    if config is not None and getattr(config, "embedding_path", None):
        emb_path = getattr(config, "test_embedding_path", None) if test else None
        if emb_path is None:
            emb_path = config.embedding_path
            if test:
                test_emb_path = Path(emb_path).with_name("embeddings-test.pt")
                if test_emb_path.is_file():
                    emb_path = test_emb_path
        emb = torch.load(emb_path, map_location='cpu')
        if norm:
            emb /= emb.norm(dim=-1, keepdim=True)
        if not with_label:
            return emb
        label_path = getattr(config, "test_label_path", None) if test else getattr(config, "label_path", None)
        if label_path is None:
            emb_path = Path(emb_path)
            suffix = "-test" if test else ""
            label_candidates = [
                emb_path.with_name(f"label{suffix}.pt"),
                emb_path.with_name(f"labels{suffix}.pt"),
            ]
            for candidate in label_candidates:
                if candidate.is_file():
                    label_path = candidate
                    break
        if label_path is None:
            split = "test" if test else "train"
            expected = " or ".join(str(path) for path in label_candidates)
            raise ValueError(
                f"Precomputed {split} embeddings require a label file. "
                f"Expected {expected}, or pass {'--test_label_path' if test else '--label_path'}."
            )
        if not Path(label_path).is_file():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        label = torch.load(label_path, map_location='cpu')
        if len(emb) != len(label):
            raise ValueError(
                f"Embedding/label length mismatch for {'test' if test else 'train'} split: "
                f"{emb_path} has {len(emb)} rows, but {label_path} has {len(label)} labels."
            )
        return emb, label
    p, test_str = _embedding_path(arch, config, datapath, dataset, test)
    emb = torch.load(p / f'embeddings{test_str}.pt', map_location='cpu')
    if norm:
        emb /= emb.norm(dim=-1, keepdim=True)
    if not with_label:
        return emb
    label = torch.load(p / f'label{test_str}.pt', map_location='cpu')
    return emb, label


def _embedding_path(arch, config, datapath, dataset, test):
    assert bool(config) ^ bool(arch and dataset)
    if config:
        arch = config.arch
        dataset = config.dataset
    import gen_embeds
    test_str = '-test' if test else ''
    p = gen_embeds.get_outpath(arch, dataset, datapath)
    return p, test_str


def load_embed_stats(
        config=None,
        arch=None,
        dset=None,
        test=False,
        datapath='data'):
    p, test_str = _embedding_path(arch, config, datapath, dset, test)
    mean = torch.load(p / f'mean{test_str}.pt', map_location='cpu')
    std = torch.load(p / f'std{test_str}.pt', map_location='cpu')
    return mean, std
