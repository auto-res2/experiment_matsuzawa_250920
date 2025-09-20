"""Dataset download / transform utilities (HuggingFace + external archives)."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = (_THIS_DIR / ".." / "data").resolve()
_DATA_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
#                              HELPER FUNCTIONS                               #
###############################################################################

def _download_and_extract(url: str, dst: Path):
    """Download remote archive *once* then extract into *dst*."""

    import urllib.request

    filename = url.split("/")[-1]
    archive_path = _DATA_DIR / filename
    if not archive_path.exists():
        print(f"[preprocess] Downloading {url} → {archive_path} …", flush=True)
        urllib.request.urlretrieve(url, archive_path)
    print(f"[preprocess] Extracting {archive_path} …", flush=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dst)
    elif archive_path.suffix in {".tar", ".gz", ".tgz"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dst)
    else:
        raise ValueError(f"[preprocess] Unknown archive format: {archive_path}")


###############################################################################
#                                 DATASETS                                    #
###############################################################################

class HFDatasetWrapper(Dataset):
    """Wrap a `datasets.Dataset` so that torchvision transforms can be applied."""

    def __init__(self, hf_ds, transform):
        self.ds = hf_ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):  # type: ignore[override]
        sample = self.ds[idx]
        img = sample["image"].convert("RGB")
        label_key = "label" if "label" in sample else sample.get("category", -1)
        label = sample[label_key] if isinstance(label_key, str) else label_key
        return self.transform(img), torch.tensor(label, dtype=torch.long)


class FolderDataset(Dataset):
    """Generic *ImageFolder-style* dataset – expects <root>/<class>/<img>."""

    def __init__(self, root: Path, transform):
        self.samples = []
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for img_path in class_dir.rglob("*.*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                self.samples.append((img_path, class_dir.name))
        self.classes = sorted({c for _, c in self.samples})
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):  # type: ignore[override]
        img_path, cname = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), torch.tensor(self.cls2idx[cname], dtype=torch.long)


###############################################################################
#                                   MAPPINGS                                  #
###############################################################################
_MEAN_STD = {
    "cifar": ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201]),
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def _basic_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(*_MEAN_STD["imagenet"]),
        ]
    )


HF_ID_MAP = {
    "cifar10c": ("robro/cifar10-c-parquet", 32),
    "cifar100c": ("randall-lab/cifar100-c", 32),
    "imagenetc": ("ang9867/ImageNet-C", 224),
    "domainnet_painting": ("Bruece/domainnet-126-edge-image-painting", 224),
    "domainnet_sketch": ("Bruece/domainnet-126-by-class-sketch", 224),
    "domainnet_real": ("Bruece/domainnet-126-by-class-real", 224),
    "domainnet_clipart": ("Bruece/domainnet-126-edge-image-clipart", 224),
}

EXTERNAL_URLS = {
    "imagenetvidc": "https://zenodo.org/record/5081125/files/imagenet-vid-c.tar?download=1",
    "realblurc": "https://data.vision.ee.ethz.ch/cvl/DIV2K/RealBlur_J.zip",
    "lasotc": "https://cilab.nju.edu.cn/lasot_benchmark/LaSOTBenchmark.zip",
}


###############################################################################
#                               PUBLIC  API                                   #
###############################################################################

def get_loader(
    name: str,
    batch_size: int,
    split: str = "test",
    sample_cap: Optional[int] = None,
    num_workers: int = 8,
) -> DataLoader:
    """Return ``torch.utils.data.DataLoader`` for the requested dataset key."""

    name = name.lower()
    if name in HF_ID_MAP:
        hf_id, img_sz = HF_ID_MAP[name]
        ds = load_dataset(hf_id, split=split, trust_remote_code=True)
        ds = HFDatasetWrapper(ds, _basic_transform(img_sz))
    elif name in EXTERNAL_URLS:
        url = EXTERNAL_URLS[name]
        tgt = _DATA_DIR / name
        if not tgt.exists():
            _download_and_extract(url, tgt)
        img_sz = 224 if "imagenetvid" in name else 256
        ds = FolderDataset(tgt, _basic_transform(img_sz))
    else:
        raise ValueError(f"[preprocess] Unknown dataset key: {name}")

    if sample_cap is not None and len(ds) > sample_cap:
        g = torch.Generator().manual_seed(0)
        subset_idx = torch.randperm(len(ds), generator=g)[:sample_cap]
        ds = torch.utils.data.Subset(ds, subset_idx.tolist())

    print(f"[preprocess] {name}: final dataset size = {len(ds)}", flush=True)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
