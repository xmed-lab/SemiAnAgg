import torch

from .utils import get_pair_indices
from argparse import Namespace
# for type hint
from typing import Optional, Union, Tuple, Dict
from torch import Tensor, nn

# from .types import LossOutType, SimilarityType, DistanceLossType

from torch.nn import functional as F

from .utils import reduce_tensor, bha_coeff_log_prob, l2_distance, bha_coeff
import numpy as np
from options import args_parser
args = args_parser()
class LCC(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, cls_num_list, T):
        super(LCC, self).__init__()
        self.T = T
        self.label_distrib = torch.zeros(len(cls_num_list))
        for cls, count in enumerate(cls_num_list):
            self.label_distrib[cls] = max(1e-8, count)

    def forward(self, logit, y):
        tmp = torch.pow(self.label_distrib, -1 / 4).unsqueeze(0).expand((logit.shape[0], -1)).to(logit.device)
        cal_logit = torch.exp(
            logit
            - (
                    self.T
                    * tmp
            )
        )

        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))

        return loss.sum() / logit.shape[0]


class BlSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, reduction="mean"):
        super(BlSoftmaxLoss, self).__init__()
        # reduction: string. One of "none", "mean", "sum"
        label_count_array = cls_num_list
        label_count_array = np.array(label_count_array) / np.sum(label_count_array)
        adjustments = np.log(label_count_array + 1e-12)
        adjustments = torch.from_numpy(adjustments).view(1, -1)
        self.adjustments = adjustments
        self.reduction = reduction

    def forward(self, logits, target):
        logits = logits + self.adjustments.to(logits.device)
        loss = F.cross_entropy(input=logits, target=target, reduction=self.reduction)
        return loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.detach().exp()

        if self.alpha is not None:
            assert False

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def get_criterion(cls_num_list, rounds, loss_fn_name):
    if loss_fn_name == 'CE':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_fn_name == 'BSM':
        loss_fn = BlSoftmaxLoss(cls_num_list)
    elif loss_fn_name == 'focal':
        loss_fn = FocalLoss(gamma=2.0)
    elif loss_fn_name == 'CB':
        cls_num_list = np.array(cls_num_list)
        non_zero_indices = cls_num_list != 0
        beta = 0.9999
        effective_num = np.zeros_like(cls_num_list, dtype=np.float32)
        effective_num[non_zero_indices] = 1.0 - np.power(beta, cls_num_list[non_zero_indices])
        per_cls_weights = np.zeros_like(cls_num_list, dtype=np.float32)
        per_cls_weights[non_zero_indices] = (1.0 - beta) / np.array(effective_num[non_zero_indices])
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list[non_zero_indices])
        per_cls_weights = torch.FloatTensor(per_cls_weights)
        loss_fn = nn.CrossEntropyLoss(weight=per_cls_weights)
    elif loss_fn_name == 'RW':
        cls_num_list = np.array(cls_num_list)
        non_zero_indices = cls_num_list != 0
        weights = np.zeros_like(cls_num_list, dtype=np.float32)
        weights[non_zero_indices] = np.sum(cls_num_list) / (len(cls_num_list) * (cls_num_list[non_zero_indices]))
        weights = torch.FloatTensor(weights)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    elif loss_fn_name == 'LCC':
        cls_num_list = np.array(cls_num_list)
        loss_fn = LCC(cls_num_list=cls_num_list, T=0.5)

    else:

        raise Exception("this loss is not defined")

    return loss_fn


def softmax_cross_entropy_loss(logits: Tensor, targets: Tensor, dim: int = 1, reduction: str = 'mean') -> Tensor:
    """
    :param logits: (labeled_batch_size, num_classes) model output of the labeled data
    :param targets: (labeled_batch_size, num_classes) labels distribution for the data
    :param dim: the dimension or dimensions to reduce
    :param reduction: choose from 'mean', 'sum', and 'none'
    :return:
    """
    loss = -torch.sum(F.log_softmax(logits, dim=dim) * targets, dim=dim)

    return reduce_tensor(loss, reduction)


def mse_loss(prob: Tensor, targets: Tensor, reduction: str = 'mean', **kwargs) -> Tensor:
    return F.mse_loss(prob, targets, reduction=reduction)


def bha_coeff_loss(targets: Tensor, logits: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    """
    Bhattacharyya coefficient of p and q; the more similar the larger the coefficient
    :param logits: (batch_size, num_classes) model predictions of the data
    :param targets: (batch_size, num_classes) label prob distribution
    :param dim: the dimension or dimensions to reduce
    :param reduction: reduction method, choose from "sum", "mean", "none
    :return: Bhattacharyya coefficient of p and q, see https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    log_probs = F.log_softmax(logits, dim=dim)
    log_targets = torch.log(targets)

    # since BC(P,Q) is maximized when P and Q are the same, we minimize 1 - B(P,Q)
    return 1. - bha_coeff_log_prob(log_probs, log_targets, dim=dim, reduction=reduction)


def l2_dist_loss(probs: Tensor, targets: Tensor, dim: int = 1, reduction: str = "none") -> Tensor:
    loss = l2_distance(probs, targets, dim=dim)

    return reduce_tensor(loss, reduction)


class PairLoss:
    def __init__(self,
                 similarity_metric=bha_coeff,
                 distance_loss_metric=bha_coeff_loss,
                 confidence_threshold: float = 0.,
                 similarity_threshold: float = 0.9,
                 distance_use_prob: bool = True,
                 reduction: str = "mean"):
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.distance_use_prob = distance_use_prob
        # self.distance_use_prob =True

        self.reduction = reduction

        self.get_similarity = similarity_metric
        self.get_distance_loss = distance_loss_metric

    def __call__(self,
                 logits: Tensor,
                 probs: Tensor,
                 targets: Tensor,
                 true_targets: Tensor=None,
                 indices: Optional[Tensor] = None):
        """
        Args:
            logits: (batch_size, num_classes) predictions of batch data
            probs: (batch_size, num_classes) softmax probs logits
            targets: (batch_size, num_classes) one hot labels
            true_targets: (batch_size, num_classes) one hot ground truth labels; used for visualization only
        Returns: None if no pair satisfy the constraints
        """
        if indices is None:
            indices = get_pair_indices(targets, ordered_pair=True)  # [i,j], i=[0:bs], j=[0:bs], i!=j
        total_size = len(indices) // 2

        i_indices, j_indices = indices[:, 0], indices[:, 1]

        logits_j = logits[j_indices]
        probs_j = probs[j_indices]
        targets_i = targets[i_indices]
        targets_j = targets[j_indices]

        targets_max_prob = targets.max(dim=1)[0]
        targets_i_max_prob = targets_max_prob[i_indices]

        conf_mask = targets_i_max_prob > self.confidence_threshold
        # 为什么下方sim要衡量target不同组合的相似度？
        sim: Tensor = self.get_similarity(targets_i, targets_j, dim=1)
        # 修改如下：
        # sim: Tensor = self.get_similarity(targets_i, probs_j, dim=1)
        # 保留置信度高的 sharpened weak aug data output, 及与其similaity高的strong aug data output
        factor = conf_mask.float() * torch.threshold(sim, self.similarity_threshold, 0)

        if self.distance_use_prob:
            loss_input = probs_j
        else:
            loss_input = logits_j
        # 如果用bha，第一个参数处理中自带softmax
        distance_ij = self.get_distance_loss(targets_i, loss_input, dim=1, reduction='none')

        loss_mat = factor * distance_ij
        if args.test:
            selected_pair=indices[[i for i in range(loss_mat.shape[0]) if loss_mat.detach().cpu().numpy()[i]!=0]]
            match_gt=true_targets[selected_pair].squeeze().numpy()
            match_mat=compute_pred_matrix(match_gt[:,0],match_gt[:,1])

        if self.reduction == "mean":
            loss = torch.sum(loss_mat) / total_size
        elif self.reduction == "sum":
            loss = torch.sum(loss_mat)

        if args.test:
            return loss,match_mat
        else:
            return loss,{"log": {}, "plot": {}}


class UnsupervisedLoss:
    def __init__(self,
                 loss_type: str,
                 loss_thresholded: bool = False,
                 confidence_threshold: float = 0.,
                 reduction: str = "mean"):
        if loss_type in ["entropy", "cross entropy"]:
            self.loss_use_prob = False
            self.loss_fn = softmax_cross_entropy_loss
        else:
            self.loss_use_prob = True
            self.loss_fn = mse_loss

        self.loss_thresholded = loss_thresholded
        self.confidence_threshold = confidence_threshold
        self.reduction = reduction

    def __call__(self, logits: Tensor, probs: Tensor, targets: Tensor) -> Tensor:
        """
               Args:
                   logits: (unlabeled_batch_size, num_classes) model output for unlabeled data
                   targets: (unlabeled_batch_size, num_classes) guessed labels distribution for unlabeled data
        """
        loss_input = probs if self.loss_use_prob else logits
        loss = self.loss_fn(loss_input, targets, dim=1, reduction="none")

        if self.loss_thresholded:
            targets_mask = (targets.max(dim=1)[0] > self.confidence_threshold)

            if len(loss.shape) > 1:
                # mse_loss returns a matrix, need to reshape mask
                targets_mask = targets_mask.view(-1, 1)

            loss *= targets_mask.float()

        return reduce_tensor(loss, reduction=self.reduction)


def get_distance_loss_metric(distance_loss_type: str):
    # other distance loss functions can be added here
    if distance_loss_type == "l2":
        distance_use_prob = True
        distance_loss_metric = l2_dist_loss

    else:
        distance_use_prob = False
        distance_loss_metric = bha_coeff_loss

    return distance_loss_metric, distance_use_prob


def build_pair_loss(args: Namespace, reduction: str = "mean") -> PairLoss:
    # similarity_metric = get_similarity_metric(args.similarity_type)
    distance_loss_metric, distance_use_prob = get_distance_loss_metric(args.distance_loss_type)

    return PairLoss(
        similarity_metric=bha_coeff,
        distance_loss_metric=distance_loss_metric,
        confidence_threshold=args.confidence_threshold,
        similarity_threshold=args.similarity_threshold,
        distance_use_prob=distance_use_prob,
        reduction=reduction)

def compute_pred_matrix(gt, pred):
    matrix=np.zeros([10,10])
    for idx_gt in range(len(gt)):
        matrix[int(gt[idx_gt])][int(pred[idx_gt])]+=1
    return matrix
