"""Channel-independent dataset wrapper for univariate-per-feature iteration.

Implements the "channel-independent" data arrangement from the Time-LLM
reference project, where each feature channel of a multivariate time-series
is treated as a separate univariate sample.

Given a base dataset that returns samples of shape ``(T, C)``, this wrapper
produces ``C`` samples per original sample, each of shape ``(T, 1)``.  The
approach is fully transparent to downstream models — they see batches of
univariate series without knowing the original multivariate structure.

Usage::

    from liulian.data.ts.channel_independent import ChannelIndependentDataset

    base_split = dataset.get_split('train')   # TimeSeriesSplit
    ci_split = ChannelIndependentDataset(base_split)
    # len(ci_split) == len(base_split) * n_features
    # ci_split[i] returns (feat, target, time) with feat.shape == (T, 1)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class ChannelIndependentDataset(Dataset):
    """Wraps a time-series dataset to iterate per feature channel.

    Each base sample ``(T, C_feat)`` with target ``(T, C_targ)`` becomes
    ``C_feat`` univariate samples.  Target channels are broadcast: each
    feature-derived sample keeps all target channels (or just the
    corresponding one if ``match_target_channel=True``).

    Attributes:
        base: The underlying dataset (e.g. :class:`TimeSeriesSplit`).
        n_channels: Number of feature channels in the base dataset.
    """

    def __init__(
        self,
        base: Dataset,
        n_channels: int | None = None,
        match_target_channel: bool = False,
    ) -> None:
        """Initialize the channel-independent wrapper.

        Args:
            base: Base dataset returning ``(feat, target, time)`` tuples.
            n_channels: Number of feature channels.  If *None*, inferred
                from the first sample.
            match_target_channel: When *True* and target has the same
                number of channels as features, each derived sample uses
                only its corresponding target channel.  Otherwise all
                target channels are kept.
        """
        super().__init__()
        self.base = base
        self.match_target_channel = match_target_channel

        if n_channels is None:
            sample = base[0]
            feat = sample[0] if isinstance(sample, (tuple, list)) else sample
            n_channels = feat.shape[-1] if feat.ndim >= 2 else 1
        self.n_channels = n_channels

    def __len__(self) -> int:
        return len(self.base) * self.n_channels

    def __getitem__(self, idx: int):
        """Return the *idx*-th channel-independent sample.

        Returns:
            Tuple ``(feat, target, time)`` where ``feat`` has shape
            ``(T, 1)`` (single channel).
        """
        base_idx = idx // self.n_channels
        channel_idx = idx % self.n_channels

        sample = self.base[base_idx]

        if isinstance(sample, (tuple, list)):
            feat, target, time = sample[0], sample[1], sample[2]
        else:
            raise TypeError(
                f'Expected (feat, target, time) tuple, got {type(sample)}'
            )

        # Extract single feature channel
        if feat.ndim >= 2:
            feat = feat[:, channel_idx: channel_idx + 1]
        else:
            feat = feat.unsqueeze(-1)

        # Optionally match target channel
        if (
            self.match_target_channel
            and target.ndim >= 2
            and target.shape[-1] == self.n_channels
        ):
            target = target[:, channel_idx: channel_idx + 1]

        return feat, target, time
