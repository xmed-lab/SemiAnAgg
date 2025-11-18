import logging
import os
import sys
import time
import torch
from torch.optim.lr_scheduler import MultiStepLR

from cifar_load import get_dataloader

# from moco_pytorch import model

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

from torch.nn import functional as F
from tqdm import tqdm

from utils.metrics import compute_metrics_test


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lc_learning_rate
    for milestone in args.lc_schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# test using a knn monitor
def knn_test(args, net, X_train, y_train, X_test, y_test, num_classes, device=None):
    memory_data_loader, train_ds = get_dataloader(args,
                                                  X_train,
                                                  y_train,
                                                  args.dataset,
                                                  args.datadir, args.batch_size, is_labeled=True, is_testing=True)

    test_data_loader, test_ds = get_dataloader(args, X_test, y_test,
                                               args.dataset, args.datadir, args.batch_size,
                                               is_labeled=True, is_testing=True)

    net.to(device)
    net.eval()

    classes = num_classes
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        target_list = []
        train_bar = tqdm(memory_data_loader)

        for study, images, target in train_bar:
            feature, _ = net(images.to(device, non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target = target.long().squeeze()
            target_list = target_list + target.tolist()
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # print('feature_bank_device', feature_bank.device)
        feature_labels = torch.tensor(target_list, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for study, data, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, _ = net(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)

            total_num += data.size(0)
            target = target.long().squeeze()
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


import numpy as np


def inference(loader, model, device,psuedo=False):
    feature_vector = []
    labels_vector = []
    model.eval()
    for step, (_, x, y) in enumerate(loader):
        y = y.long().squeeze()
        if len(y.shape) == 0:
            continue
        if psuedo:
            x = x[0].to(device)
        else:
            x = x.to(device)

        # get encoding
        with torch.no_grad():
            h, _ = model(x)

        h = h.squeeze()
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())

        labels_vector.extend(y.numpy())

        # if step % 5 == 0:
        #     print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(model, train_loader, test_loader, device,psuedo=False):
    train_X, train_y = inference(train_loader, model, device,psuedo)
    test_X, test_y = inference(test_loader, model, device,psuedo)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size, return_index=False):
    if return_index:
        train = torch.utils.data.TensorDataset(
            torch.from_numpy(np.arange(len(X_train))), torch.from_numpy(X_train), torch.from_numpy(y_train)
        )

        test = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )
    else:
        train = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )

        test = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


from collections import defaultdict


def test_result(test_loader, logreg, device, n_classes):
    # Test fine-tuned model
    # logger.info("### Calculating final testing performance ###")
    logreg.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)

    gt_study = {}
    pred_study = {}
    studies = []
    i = 0
    for step, (h, y) in enumerate(test_loader):
        h = h.to(device)
        y = y.to(device)

        outputs = logreg(h)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, y, topk=(1, 5))

        top1.update(acc1[0], h.size(0))
        top5.update(acc5[0], h.size(0))

        output = F.softmax(outputs, dim=1)

        for j in range(len(y)):
            gt_study[i] = y[j]
            pred_study[i] = output[j]
            studies.append(i)
            i += 1

    for study in studies:
        gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
        pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)

    AUROCs, Accus, Pre, Recall, b_Accus = compute_metrics_test(gt, pred, n_classes=n_classes)

    return top1.avg, b_Accus, AUROCs, Accus, Pre, Recall


def test_fed_ssl(args, encoder_model, logreg, X_train, y_train, X_test, y_test, num_classes=None,
                 device='cuda'):
    memory_data_loader, train_ds = get_dataloader(args,
                                                  X_train,
                                                  y_train,
                                                  args.dataset,
                                                  args.datadir, args.batch_size, is_labeled=True, is_testing=True)

    test_data_loader, test_ds = get_dataloader(args, X_test, y_test,
                                               args.dataset, args.datadir, args.batch_size,
                                               is_labeled=True, is_testing=True)

    print("Creating features from pre-trained model")
    encoder_model.to(device)
    (train_X, train_y, test_X, test_y) = get_features(
        encoder_model, memory_data_loader, test_data_loader, device
    )

    train_loader, test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, 256
    )

    # fine-tune model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=logreg.parameters(), lr=0.1, weight_decay=0, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)

    best_acc1, best_bAccus_avg, best_AUROCs, best_Accus, best_Pre, best_Recall = 0, 0, 0, 0, 0, 0
    # Train fine-tuned model
    logreg.train()
    for epoch in range(100):
        metrics = defaultdict(list)
        for step, (h, y) in enumerate(train_loader):
            h = h.to(device)
            y = y.to(device)

            outputs = logreg(h)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy and save metrics
            # accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            # metrics["Loss/train"].append(loss.item())
            # metrics["Accuracy/train"].append(accuracy)

        if epoch % 20 == 0:
            # print(logreg.training)
            acc1, bAccus_avg, AUROC_avg, Accus_avg, Pre, Recall = test_result(test_loader, logreg, device, num_classes,
                                                                              )
            print("======epoch {}==acc: {}, wacc:{}====".format(epoch, acc1, bAccus_avg))

            if bAccus_avg > best_bAccus_avg:
                best_acc1, best_bAccus_avg, best_AUROCs, best_Accus, best_Pre, best_Recall = acc1, bAccus_avg, AUROC_avg, Accus_avg, Pre, Recall

            logreg.train()
        scheduler.step()

    acc1, bAccus_avg, AUROC_avg, Accus_avg, Pre, Recall = test_result(test_loader, logreg, device, num_classes
                                                                      )

    if bAccus_avg > best_bAccus_avg:
        best_acc1, best_bAccus_avg, best_AUROCs, best_Accus, best_Pre, best_Recall = acc1, bAccus_avg, AUROC_avg, Accus_avg, Pre, Recall

    return best_AUROCs, best_Accus, best_Pre, best_Recall, best_bAccus_avg
