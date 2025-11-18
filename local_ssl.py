from functools import partial

from sklearn.metrics._classification import balanced_accuracy_score
from torch.utils.data import Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
import numpy as np

from lncs import get_features
from networks.resnet import resnet18
from networks.resnetcifar import ResNet18_cifar10
from networks.ssl_models import get_model
from options import args_parser
# from networks.models import ModelFedCon
from utils import losses, ramps
import torch.nn as nn

from utils.lars import LARS, static_lr, remove_bias_and_norm_from_weight_decay
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.utils_SimPLE import label_guessing, sharpen, get_class_compose, adjust_learning_rate
from loss.loss import UnsupervisedLoss  # , build_pair_loss
import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.setLevel(logging.INFO)

from torchvision import transforms
from utils.ramp import LinearRampUp
from timm.scheduler.cosine_lr import CosineLRScheduler

args = args_parser()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# alpha=0.999
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, weak_aug, strong_aug, label = self.dataset[self.idxs[item]]
        return items, index, weak_aug, strong_aug, label


class SelfsupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes, add_cls_to_optim=True):
        # net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
        # net_ema = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
        self.n_classes = n_classes
        net = get_model(args, n_classes)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])

        self.model = net.cuda()

        self.data_idxs = idxs
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.base_lr = args.base_lr
        self.softmax = nn.Softmax()
        self.max_grad_norm = args.max_grad_norm
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        online_classifier: nn.Module = nn.Linear(self.model.backbone.fc.in_features, n_classes)
        self.online_classifier = online_classifier.cuda()

        self.max_step = args.rounds * round(len(self.data_idxs) / args.batch_size)
        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.base_lr,
                                              betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            if add_cls_to_optim:
                learnable_params = [
                    {"name": "backbone", "params": self.model.parameters()},
                    {
                        "name": "classifier",
                        "params": self.online_classifier.parameters(),
                        "lr": 0.1 if args.warmup else args.base_lr,
                        "weight_decay": 0,

                    },
                ]

            else:
                learnable_params = [
                    {"name": "backbone", "params": self.model.parameters()},
                ]

            self.optimizer = torch.optim.SGD(learnable_params, lr=args.base_lr, momentum=0.9,
                                             weight_decay=5e-4)
            if args.timm_cos:
                self.scheduler = CosineLRScheduler(self.optimizer, t_initial=args.rounds * args.local_ep,
                                                   warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)

        elif args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.base_lr,
                                               weight_decay=0.02)

        elif args.opt == 'lars':
            scale_factor = args.batch_size / 256
            learnable_params = [
                {"name": "backbone", "params": self.model.parameters()},
                {
                    "name": "classifier",
                    "params": self.online_classifier.parameters(),
                    "lr": 0.1,
                    "weight_decay": 0,
                },
            ]
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)
            idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]

            self.optimizer = LARS(learnable_params, lr=args.base_lr * scale_factor, weight_decay=1e-4,
                                  clip_lr=True,
                                  eta=0.02,
                                  exclude_bias_n_norm=True, momentum=0.9)
            self.scheduler = LinearWarmupCosineAnnealingLR(
                self.optimizer,
                warmup_epochs=10,
                max_epochs=args.rounds * args.local_ep,
                warmup_start_lr=0.00003,
                eta_min=0,
            )

            if idxs_no_scheduler:
                partial_fn = partial(
                    static_lr,
                    get_lr=self.scheduler.get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[args.base_lr * scale_factor] * len(idxs_no_scheduler),
                )
                self.scheduler.get_lr = partial_fn

        self.max_warmup_step = round(len(self.data_idxs) / args.batch_size) * args.num_warmup_epochs
        self.ramp_up = LinearRampUp(length=self.max_warmup_step)

    def train(self, args, net_w, net_cls, op_dict, epoch, unlabeled_idx, train_dl_local, n_classes, loss_fn=None,
              train_classifier=False, fix_match=False, class_confident=None, client_idx=None,train_dl_local_unl=None):

        if epoch == 0:
            if args.random_ssl:
                data_ssl_prefix = 'data_rand'
            else:
                data_ssl_prefix = 'data_ssl'

            if args.num_users != 10 or args.beta != 0.8:
                data_ssl_prefix = f'{data_ssl_prefix}_{args.num_users}_{args.beta}'
            else:
                data_ssl_prefix = data_ssl_prefix

            if args.long_tailed:
                self.train_X_ssl = torch.load(f'{data_ssl_prefix}/{args.dataset}_LT/{unlabeled_idx}.pt').cuda()
            else:
                self.train_X_ssl = torch.load(f'{data_ssl_prefix}/{args.dataset}/{unlabeled_idx}.pt').cuda()
            print('Mean:', torch.mean(self.train_X_ssl))

        if args.ssl_model == 'byol':
            pretrained_dict = {k: v for k, v in net_w.items() if not k.startswith('target_encoder')}
            msg = self.model.load_state_dict(copy.deepcopy(pretrained_dict), strict=False)
            assert all(l.startswith('target_encoder') for l in msg.missing_keys)
        else:
            self.model.load_state_dict(copy.deepcopy(net_w))


        # class_cos_ssl = torch.zeros(n_classes, 1).cuda()
        class_cos_ssl = torch.ones(n_classes, 1).cuda()

        self.model.cuda()
        self.model.eval()

        (train_X, train_y, _, _) = get_features(
            self.model.backbone, train_dl_local, [], device='cuda',psuedo=True
        )
        train_y = torch.from_numpy(train_y).cuda()
        train_X = torch.from_numpy(train_X).cuda()


        self.model.train()
        self.online_classifier.load_state_dict(net_cls)

        if train_classifier or args.warmup:
            # logger.info('classifier into supervised client')
            self.online_classifier.train()
        else:
            self.online_classifier.eval()

        self.online_classifier.cuda()
        self.model.cuda()

        if args.opt == 'lars' or args.timm_cos:
            self.optimizer.load_state_dict(op_dict['optimizer'])
            self.scheduler.load_state_dict(op_dict['scheduler'])
        else:
            self.optimizer.load_state_dict(op_dict)
            # adjust_learning_rate(self.optimizer, epoch, args)
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = self.unsup_lr

        self.epoch = epoch

        epoch_loss = []
        epoch_loss_clr = []
        logger.info('Unlabeled client %d begin self-supervised training' % unlabeled_idx)
        correct_pseu = 0
        all_pseu = 0
        test_right = 0
        test_right_ema = 0
        train_right = 0
        same_total = 0
        length_of_trainloader = len(train_dl_local)

        class_centeroids = torch.zeros(self.n_classes, 512).cuda()
        class_count = torch.tensor(np.array(get_class_compose(train_dl_local, self.n_classes))).cuda()
        num = 0
        selected_label = torch.ones((len(train_dl_local.dataset),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda()
        client_used_psuedo_local = []
        for epoch in range(args.local_ep):
            batch_loss = []
            batch_loss_cls = []
            for i, (x_ulb_idx, weak_aug_batch, label_batch) in enumerate(train_dl_local):
                if len(label_batch) == 1:
                    continue

                if weak_aug_batch[0].size(0) == 1:
                    continue
                weak_aug_batch = [weak_aug_batch[version].cuda() for version in range(len(weak_aug_batch))]

                x_ulb_idx = x_ulb_idx.cuda().long().squeeze()

                if len(x_ulb_idx.shape) == 0:
                    x_ulb_idx = x_ulb_idx.unsqueeze(dim=0)

                label_batch = label_batch.cuda()
                label_batch = label_batch.long().squeeze()

                if len(label_batch.shape) == 0:
                    label_batch = label_batch.unsqueeze(dim=0)

                p1, p2, loss_u = self.model(weak_aug_batch[0], weak_aug_batch[1])

                if train_classifier:
                    outputs = self.online_classifier(p1)

                    outputs2 = self.online_classifier(p2)

                else:
                    outputs = self.online_classifier(p1.detach())

                    outputs2 = self.online_classifier(p2.detach())

                if len(outputs.shape) != 2:
                    outputs = outputs.unsqueeze(dim=0)
                    outputs2 = outputs2.unsqueeze(dim=0)

                if fix_match:
                    sharpened = sharpen(outputs)
                    loss_classification = torch.sum(losses.softmax_mse_loss(outputs2, sharpened)) / args.batch_size

                else:
                    # logger.info('classifier into supervised client')
                    loss_classification = (loss_fn(outputs, label_batch) + loss_fn(outputs2, label_batch)) / 2
                    # loss_classification = (loss_fn(outputs, label_batch) + loss_fn(outputs2, label_batch)) / 2
                    # loss_classification = loss_fn(outputs, label_batch)

                # ramp_up_value = self.ramp_up(current=self.epoch)

                # loss = ramp_up_value * args.lambda_u * loss_u

                loss = loss_u * args.scale_loss + loss_classification

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                # lr_ = optimizer.param_groups[0]['lr']

                batch_loss.append(loss_u.item())

                batch_loss_cls.append(loss_classification.item())

                pseudo_label = torch.softmax(outputs, dim=-1)
                max_probs, max_idx = torch.max(pseudo_label, dim=-1)
                select = max_probs.ge(args.main_T).long()

                if x_ulb_idx[select == 1].nelement() != 0:
                    selected_label[x_ulb_idx[select == 1]] = max_idx[select == 1]
                    client_used_psuedo_local.extend(x_ulb_idx[select == 1].tolist())

                self.iter_num = self.iter_num + 1
                if args.vis_ph:
                    labels = label_batch.view(label_batch.size(0), 1).expand(-1, p1.size(1)).cuda()
                    semantic_proj = nn.functional.normalize(p1, dim=-1)  # already normalized
                    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
                    batch_unique_target = unique_labels[:, 0].long()
                    res = torch.zeros_like(class_centeroids, dtype=torch.float).scatter_add_(0,
                                                                                             labels.type(torch.int64),
                                                                                             semantic_proj.detach())
                    res[batch_unique_target] = res[batch_unique_target] / labels_count.float().unsqueeze(1)
                    target_class_total_count = class_count[batch_unique_target]

                    class_centeroids[batch_unique_target] += (labels_count / target_class_total_count).unsqueeze(1) * \
                                                             res[batch_unique_target]

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_loss_clr.append(sum(batch_loss_cls) / len(batch_loss))

            self.epoch = self.epoch + 1

        pl_bacc = balanced_accuracy_score(train_dl_local.dataset.labels[client_used_psuedo_local],
                                          selected_label[client_used_psuedo_local].cpu().detach().numpy())
        logger.info(
            f'selected number {len(client_used_psuedo_local)} accuracy of selected number {pl_bacc}')

        chosen_sup = nn.functional.normalize(train_X[client_used_psuedo_local],
                                             dim=-1)  # already normalized
        chosen_sup_ssl = self.train_X_ssl[client_used_psuedo_local]

        cosine_sim = torch.cosine_similarity(chosen_sup,
                                             chosen_sup_ssl, dim=-1).unsqueeze(1).cuda()

        psuedo_labels = train_y[client_used_psuedo_local].view(train_y[client_used_psuedo_local].size(0), 1).expand(-1, 1)
        unique_psuedo_labels, psuedo_labels_count = psuedo_labels.unique(dim=0, return_counts=True)

        psuedo_batch_unique_target = unique_psuedo_labels[:, 0].long()
        res = torch.zeros_like(class_cos_ssl, dtype=torch.float).scatter_add_(0,
                                                                              psuedo_labels.type(
                                                                                  torch.int64),
                                                                              cosine_sim.detach())

        res[psuedo_batch_unique_target] = res[psuedo_batch_unique_target] / psuedo_labels_count.float().unsqueeze(1)

        class_cos_ssl[psuedo_batch_unique_target] = res[psuedo_batch_unique_target]


        if args.vis_ph:
            class_centeroids = nn.functional.normalize(class_centeroids, dim=-1)
            class_centeroids = class_centeroids.detach()

        if args.vis_collapse:
            chosen_sup = nn.functional.normalize(train_X,
                                                 dim=-1)  # already normalized
            chosen_sup_ssl = self.train_X_ssl

            cosine_sim = torch.cosine_similarity(chosen_sup,
                                                 chosen_sup_ssl, dim=-1).unsqueeze(1).cuda()

            psuedo_labels = train_y.view(train_y.size(0), 1).expand(-1, 1)
            unique_psuedo_labels, psuedo_labels_count = psuedo_labels.unique(dim=0, return_counts=True)

            psuedo_batch_unique_target = unique_psuedo_labels[:, 0].long()
            res = torch.zeros_like(class_cos_ssl, dtype=torch.float).scatter_add_(0,
                                                                                  psuedo_labels.type(
                                                                                      torch.int64),
                                                                                  cosine_sim.detach())

            res[psuedo_batch_unique_target] = res[psuedo_batch_unique_target] / psuedo_labels_count.float().unsqueeze(1)

            class_cos_ssl[psuedo_batch_unique_target] = res[psuedo_batch_unique_target]


        self.model.cpu()
        self.online_classifier.cpu()

        if args.opt == 'lars' or args.timm_cos:
            self.scheduler.step(self.epoch)
            opt = {"optimizer": copy.deepcopy(self.optimizer.state_dict()),
                   "scheduler": copy.deepcopy(self.scheduler.state_dict())}
        else:
            opt = copy.deepcopy(
                self.optimizer.state_dict())

        model_dict = self.model.state_dict()

        if args.ssl_model == 'byol':
            model_dict = {k: v for k, v in model_dict.items() if not k.startswith('target_encoder')}

        return model_dict, self.online_classifier.state_dict(), sum(epoch_loss) / len(
            epoch_loss), sum(epoch_loss_clr) / len(
            epoch_loss), opt, class_centeroids, class_count, 0, 0, class_cos_ssl, pl_bacc,class_cos_ssl
