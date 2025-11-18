import math

import torch
from functools import reduce
from typing import Sequence, Tuple
from random import random
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3, lamb=0.8):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.lamb = lamb
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)
        self.beta = beta

    def forward(self, index, output, label):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
                (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lamb * elr_reg
        return final_loss


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.base_lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / (args.rounds * args.local_ep)))
    # else:  # stepwise lr schedule
    #     for milestone in args.schedule:
    #         lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_class_compose(train_dataloader, num_classes):
    class_count_per_client = [0] * (num_classes)
    class_idx, class_count_per_client_tr = np.unique(train_dataloader.dataset.labels,
                                                     return_counts=True)
    for idx, j in enumerate(class_idx):
        class_count_per_client[j] = class_count_per_client_tr[idx]

    return class_count_per_client


def label_guessing(model: nn.Module, batches_1: Sequence[Tensor], model_type=None) -> Tensor:
    model.eval()
    with torch.no_grad():
        probs = [F.softmax(model(batch)[-1], dim=1) for batch in batches_1]
        mean_prob = reduce(lambda x, y: x + y, probs) / len(batches_1)

    return mean_prob


def sharpen(x: Tensor, t=0.5) -> Tensor:
    sharpened_x = x ** (1 / t)
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)


class RandomAugmentation(nn.Module):
    def __init__(self, augmentation: nn.Module, p: float = 0.5, same_on_batch: bool = False):
        super().__init__()

        self.prob = p
        self.augmentation = augmentation
        self.same_on_batch = same_on_batch

    def forward(self, images: Tensor) -> Tensor:
        is_batch = len(images) < 4

        if not is_batch or self.same_on_batch:
            if random() <= self.prob:
                out = self.augmentation(images)
            else:
                out = images
        else:
            out = self.augmentation(images)
            batch_size = len(images)

            # get the indices of data which shouldn't apply augmentation
            indices = torch.where(torch.rand(batch_size) > self.prob)
            out[indices] = images[indices]

        return out
