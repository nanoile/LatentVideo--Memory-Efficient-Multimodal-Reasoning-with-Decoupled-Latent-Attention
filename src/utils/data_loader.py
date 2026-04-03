import gc
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path


class DummyVideoDataset(Dataset):
    """Dummy dataset for testing distillation pipeline.

    Cached-feature mode only returns paths in __getitem__; teacher .pt can instead be
    loaded via CachedFeatureDataset + collate_cached_feature_batch (recommended).
    """

    def __init__(self, num_samples=1000, seq_len=10, hidden_size=4096, feature_dir=None, max_seq_len=None):
        self.num_samples = num_samples
        self.seq_len = int(seq_len if max_seq_len is None else min(seq_len, max_seq_len))
        self.hidden_size = hidden_size
        self.feature_dir = Path(feature_dir).resolve() if feature_dir else None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with torch.no_grad():
            video_embeds = torch.randn(self.seq_len, self.hidden_size, dtype=torch.float32).detach()
        result = {"inputs_embeds": video_embeds}

        if self.feature_dir:
            result["feature_path"] = str(self.feature_dir / f"sample_{idx:06d}.pt")

        return result


def collate_cached_feature_batch(batch):
    """Stack teacher caches produced by CachedFeatureDataset (CPU tensors)."""
    inputs_embeds = torch.stack([b["inputs_embeds"] for b in batch])
    n_layers = len(batch[0]["teacher_hiddens"])
    teacher_hidden_batch = [
        torch.stack([b["teacher_hiddens"][j] for b in batch])
        for j in range(n_layers)
    ]
    teacher_logits_batch = torch.stack([b["teacher_logits"] for b in batch])
    return {
        "inputs_embeds": inputs_embeds,
        "teacher_hidden_batch": teacher_hidden_batch,
        "teacher_logits_batch": teacher_logits_batch,
    }


class CachedFeatureDataset(Dataset):
    """Loads one `sample_XXXXXX.pt` per __getitem__; drops pickle dict immediately to limit RAM."""

    def __init__(
        self,
        feature_dir,
        num_samples,
        seq_len,
        hidden_size,
        max_seq_len=None,
    ):
        self.feature_dir = Path(feature_dir).resolve()
        self.num_samples = int(num_samples)
        self.seq_len = int(seq_len if max_seq_len is None else min(seq_len, max_seq_len))
        self.hidden_size = int(hidden_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        path = self.feature_dir / f"sample_{idx:06d}.pt"
        L = self.seq_len
        with torch.no_grad():
            blob = None
            try:
                blob = torch.load(path, map_location="cpu", weights_only=True)
            except (TypeError, RuntimeError):
                try:
                    blob = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    blob = torch.load(path, map_location="cpu")
            h_list = blob["hidden_states"]
            logits = blob["logits"].detach()
            del blob
            teacher_hiddens = []
            for t in h_list:
                t = t.detach()
                if t.shape[0] > L:
                    t = t[:L].contiguous()
                teacher_hiddens.append(t)
            del h_list
            if logits.shape[0] > L:
                logits = logits[:L].contiguous()
            inputs_embeds = torch.randn(L, self.hidden_size, dtype=torch.float32).detach()
        return {
            "inputs_embeds": inputs_embeds,
            "teacher_hiddens": teacher_hiddens,
            "teacher_logits": logits,
        }


def create_cached_feature_dataloader(
    feature_dir,
    num_samples,
    batch_size,
    hidden_size,
    seq_len=10,
    max_seq_len=None,
    distributed=False,
):
    """DataLoader for phase3_distillation_cached: load .pt inside Dataset, num_workers=0, no pin_memory."""
    dataset = CachedFeatureDataset(
        feature_dir=feature_dir,
        num_samples=num_samples,
        seq_len=seq_len,
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
    )

    use_dist = (
        distributed
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )
    sampler = DistributedSampler(dataset, shuffle=True) if use_dist else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=collate_cached_feature_batch,
    )


def create_dataloader(
    dataset_name,
    batch_size=2,
    num_samples=1000,
    num_workers=4,
    feature_dir=None,
    max_seq_len=None,
    distributed=False,
    hidden_size=None,
    seq_len=None,
):
    """Create dataloader for distillation.

    Args:
        dataset_name: 'webvid' or 'panda-70m' (currently uses dummy data)
        batch_size: Batch size per GPU
        num_samples: Total number of samples
        num_workers: Number of data loading workers
        feature_dir: Directory containing cached teacher features (optional)
        max_seq_len: Cap dummy sequence length (limits VRAM spikes on long sequences)
        distributed: If True and torch.distributed is initialized, use DistributedSampler
        hidden_size: Dummy inputs_embeds last dim (default 4096; use model_config for alignment)
        seq_len: Dummy sequence length (default 10)
    """
    _hs = hidden_size if hidden_size is not None else 4096
    _sl = seq_len if seq_len is not None else 10
    dataset = DummyVideoDataset(
        num_samples=num_samples,
        seq_len=_sl,
        hidden_size=_hs,
        feature_dir=feature_dir,
        max_seq_len=max_seq_len,
    )

    use_dist = (
        distributed
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )
    sampler = DistributedSampler(dataset, shuffle=True) if use_dist else None

    pin_memory = feature_dir is None

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataloader
