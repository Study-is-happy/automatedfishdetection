# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

__all__ = ["subsample_labels"]


def subsample_labels(labels, num_samples, positive_fraction, bg_label):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = torch.nonzero(
        (labels != -1) & (labels != 0) & (labels != bg_label)).squeeze(1)

    background = torch.nonzero(labels == bg_label).squeeze(1)
    negative = torch.nonzero(labels == 0).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)

    num_bg = num_samples - num_pos
    num_bg = min(background.numel(), num_bg)

    num_neg = num_samples - num_pos - num_bg
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(
        background.numel(), device=background.device)[:num_bg]
    perm3 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    bg_idx = background[perm2]
    neg_idx = negative[perm3]

    # print(num_pos)
    # print(num_bg)
    # print(num_neg)
    # print()

    return pos_idx, torch.cat((bg_idx, neg_idx))
