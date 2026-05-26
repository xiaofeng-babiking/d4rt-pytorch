"""Custom collate function for D4RT batches.

Handles the nested `targets` dict produced by `PointOdysseyDataset.__getitem__`.
"""

import torch


def collate_fn(batch):
    """Stack a list of sample dicts into a batched dict.

    Each sample is a dict with top-level tensors and one nested `targets` dict
    of tensors. Top-level tensors are stacked along dim 0; nested target
    tensors are stacked the same way under the `targets` key.
    """
    out = {}
    for k in batch[0]:
        if k == "targets":
            out[k] = {
                tk: torch.stack([b["targets"][tk] for b in batch])
                for tk in batch[0]["targets"]
            }
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
