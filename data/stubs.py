"""Placeholder dataset classes.

These exist to satisfy `train.py`'s imports. They raise NotImplementedError on
construction with a clear pointer to PointOdyssey, which is the fully
implemented dataset for now.
"""


class _StubDataset:
    def __init__(self, data_root, split='train', num_frames=48, img_size=256,
                 num_queries=2048, transform=None, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is not implemented yet. "
            f"Use --dataset pointodyssey. "
            f"See docs/superpowers/specs/2026-05-25-data-module-design.md"
        )


class VideoDataset(_StubDataset):
    pass


class KubricDataset(_StubDataset):
    pass


class SintelDataset(_StubDataset):
    pass


class ScanNetDataset(_StubDataset):
    pass
