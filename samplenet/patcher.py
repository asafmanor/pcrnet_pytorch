import torch
import torch.nn as nn

import samplenet.ops as ops


class Patcher(nn.Module):
    """Gathers local patches of the input point cloud.

    Args:
        mode: 'knn', 'ball' - mode of gathering operation. Defaults to 'knn'.
        k (optional): Patch size. Defaults to 8.
        r (optional): Radius in the query ball mode. defaults to 0.1.

    Inputs:
        p: Reference point cloud of shape [batch_size, dim, num_point].
        q (optional): Query Point cloud of shape [batch_size, dim, num_query].
           Defaults to None. It that case, it is set to p.
        use_last (optional): Use last computed patche indices. Defaults to False.
        normalize_shift (optional): Normalize patches offset. Defaults to True.
        normalize_scale (optional): Normalize patches scale. Defaults to True.
        normalize_method (optional): Patches offset normalization method.
            see ops.normalize_group().

    Returns:
        Grouped point cloud tensor of shape [batch_size, dim, num_query, k]
    """

    def __init__(
        self,
        k: int,
        mode: str = "knn",
        r: float = 0.1,
        normalize_shift: bool = True,
        normalize_scale: bool = True,
        normalize_method: str = "mean",
    ):
        super().__init__()
        if mode not in ["ball", "knn"]:
            raise ValueError(f"unknown query mode: {self._mode}")

        self._mode, self._k, self._r = mode, k, r
        self._normalize_shift, self._normalize_scale, self._normalize_method = (
            normalize_shift,
            normalize_scale,
            normalize_method,
        )

        # Memory
        self._last_idx = None
        self._last_shift = None
        self._last_scale = None

    @staticmethod
    def group(p, idx):
        return ops.group(p, idx)

    def query(self, p, q=None):
        if q is None:
            q = p

        # idx for group [batch_size, num_query, k]
        with torch.no_grad():
            if self._mode == "knn":
                idx, _ = ops.knn_query(p, q, self._k)
            elif self._mode == "ball":
                idx = ops.ball_query(p, q, self._r, self._k)
        return idx

    def normalize(self, patches):
        """Normalize given patches of shape [batch_size, dim, num_query, k]."""
        normalized_patches, shift, scale = ops.normalize_group(
            patches,
            self._normalize_shift,
            self._normalize_scale,
            self._normalize_method,
        )
        self._last_shift = shift
        self._last_scale = scale

        return normalized_patches

    def denormalize(self, normalized_patches):
        """Inverse the last transform done by self.normalize()."""
        res = (normalized_patches * self._last_scale) + self._last_shift
        self._last_scale = None
        self._last_shift = None
        return res

    def forward(self, p, q=None, use_last=False):
        """Query and group points. Pass use_last=True to reuse the last calculated indices."""
        if q is None:
            q = p

        if use_last:
            grouped = self.group(p, self._last_idx)
        else:
            idx = self.query(p, q)  # [batch_size, num_query, k]
            self._last_idx = idx
            grouped = self.group(p, idx)  # [batch_size, dim, num_query, k]

        return grouped
