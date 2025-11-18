import pickle

from local_flex import FlexSelfsupervisedLocalUpdate
from networks.ssl_models import get_model
from options import args_parser
import logging
import sys
from timm.scheduler.cosine_lr import CosineLRScheduler

from utils.WeightHook import GaussianWeightingHook

# to init cuda before importing torch
args = args_parser()

logging.basicConfig(filename=args.tensorboard_path + '/log.txt', level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

from functools import partial

import torchvision
from torch import nn

from lncs import test_fed_ssl, knn_test
from local_ssl import SelfsupervisedLocalUpdate
from loss.loss import get_criterion
from networks.resnet import resnet18
from networks.resnetcifar import ResNet18_cifar10
import os
import wandb

from utils.lars import LARS, remove_bias_and_norm_from_weight_decay, static_lr
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR

# from torch.utils.tensorboard import SummaryWriter
from validation import epochVal_metrics_test
import random
import numpy as np
import copy
import datetime
from FedAvg import FedAvg, model_dist
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
# from networks.models import ModelFedCon
from dataloaders import dataset
from tqdm import trange
from cifar_load import get_dataloader, partition_data, partition_data_allnoniid


def split(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def test(epoch, checkpoint, online_classifier_glob_checkpoint, data_test, label_test, n_classes):
    if args.model == 'Res18':
        net = resnet18(n_classes, pretrained=args.Pretrained, KD=True)
    elif args.model == 'Res18_cifar':
        net = ResNet18_cifar10(num_classes=n_classes, KD=True)

    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
    model = net.cuda()

    if args.ssl_model == 'byol':
        msg = model.load_state_dict(checkpoint, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.fc.load_state_dict(online_classifier_glob_checkpoint)
    if args.dataset == 'SVHN' or args.dataset == 'cifar100' or args.dataset == 'cifar10':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True, pre_sz=args.pre_sz, input_sz=args.input_sz)

    AUROCs, Accus, Pre, Recall, b_Accus, class_centeroids, class_count = epochVal_metrics_test(model, test_dl,
                                                                                               args.model,
                                                                                               n_classes=n_classes,
                                                                                               return_centeroids=args.vis_ph)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()

    return AUROC_avg, Accus_avg, Pre, Recall, b_Accus, class_centeroids, class_count


if __name__ == '__main__':
    # setting of one labelled and x unlabelled
    supervised_user_id = [i for i in range(args.num_labeled)]
    unsupervised_user_id = list(range(len(supervised_user_id), args.unsup_num + len(supervised_user_id)))
    sup_num = len(supervised_user_id)
    unsup_num = len(unsupervised_user_id)
    total_num = sup_num + unsup_num

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.remove_n:
        if len(unsupervised_user_id):
            print(unsupervised_user_id)
            unsupervised_user_id.remove(args.remove_n)
            print(unsupervised_user_id)
        else:
            supervised_user_id.remove(args.remove_n)


    logger.info(str(args))
    logger.info(args.time_current)
    # writer = SummaryWriter(args.tensorboard_path)

    print('==> Reloading data partitioning strategy..')
    assert os.path.isdir('partition_strategy'), 'Error: no partition_strategy directory found!'
    train_idxs = None
    test_idxs = None
    if args.dataset == 'SVHN':
        if args.pl:
            partition = torch.load('partition_strategy_pl/SVHN_False.pth')
            net_dataidx_map = partition['labeled']
        else:
            partition = torch.load('partition_strategy/SVHN_noniid_10%labeled.pth')
            net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar100':
        if args.long_tailed:
            with open('partition_strategy/cifar100_LT_noniid_10%labeled.pth', 'rb') as f:  # Python 3: open(..., 'rb')
                net_dataidx_map, LT_idx = pickle.load(f)
            if args.pl:
                partition = torch.load('partition_strategy_pl/cifar100_True.pth')
                net_dataidx_map = partition['labeled']
        else:
            if args.pl:
                partition = torch.load('partition_strategy_pl/cifar100_False.pth')
                net_dataidx_map = partition['labeled']
            else:
                partition = torch.load('partition_strategy/cifar100_noniid_10%labeled.pth')
                net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar10':
        partition = torch.load('partition_strategy/cifar10_0.8.pth')
        net_dataidx_map = partition

    elif args.dataset == 'skin':
        if args.num_users != 10 or args.beta != 0.8:
            partition = torch.load(f'partition_strategy/skin_c_{args.num_users}_noniid_beta{args.beta}.pth')
        else:
            partition = torch.load('partition_strategy/skin_noniid_beta0.8.pth')
        net_dataidx_map = partition['data_partition']
        train_idxs = partition['train_list']
        test_idxs = partition['test_list']
        if args.pl:
            partition = torch.load('partition_strategy_pl/skin_False.pth')
            net_dataidx_map = partition['labeled']

    if args.pl:
        net_dataidx_map_un = partition['unlabeled']

    # because only load_skin needs train and test ids
    X_train, y_train, X_test, y_test, _, traindata_cls_counts = partition_data_allnoniid(
        args.dataset, args.datadir, train_idxs=train_idxs, test_idxs=test_idxs, partition=args.partition,
        n_parties=total_num, beta=args.beta)

    if args.dataset == 'SVHN':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])

    if args.long_tailed:
        X_train = X_train[LT_idx]
        y_train = y_train[LT_idx]

    if args.dataset == 'cifar10' or args.dataset == 'SVHN':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'skin':
        n_classes = 7


    # net_glob = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)

    net_glob = get_model(args, n_classes)

    # if args.model == 'Res18':
    #     # net_glob = torchvision.models.resnet18(pretrained=args.Pretrained)
    #     # net_glob.fc = nn.Linear(net_glob.fc.weight.shape[1], n_classes)
    #     net_glob = resnet18(n_classes, pretrained=args.Pretrained, KD=True)
    # elif args.model == 'Res18_cifar':
    #     net_glob = ResNet18_cifar10(num_classes=n_classes, KD=True)

    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[i for i in range(round(len(args.gpu) / 2))])  #

    net_glob.train()
    net_glob.cuda()
    net_glob(torch.randn(args.batch_size, 3, args.input_sz, args.input_sz).cuda(),
             torch.randn(args.batch_size, 3, args.input_sz, args.input_sz).cuda())
    if args.ssl_model == 'byol':
        net_glob.reset_moving_average()
    w_glob = net_glob.state_dict()
    w_locals = []
    w_ema_unsup = []
    lab_trainer_locals = []
    unlab_trainer_locals = []
    if args.remove_n:
        sup_net_locals = {}
        lab_trainer_locals = {}
        sup_classifier_locals = {}
        sup_optim_locals = {}
    else:
        lab_trainer_locals = []
        sup_classifier_locals = []
        sup_optim_locals = []

        sup_net_locals = []
    unsup_net_locals = []
    unsup_optim_locals = []
    dist_scale_f = args.dist_scale

    total_lenth = sum([len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))])
    each_lenth = [len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))]
    client_freq = [len(net_dataidx_map[i]) / total_lenth for i in range(len(net_dataidx_map))]

    wandb.init(
        # Set the project where this run will be logged
        project='SemiAnAgg',
        # Track hyperparameters and run metadata
        config=
        vars(args)
    )

    net_client_all_idx = []
    net_client_sup_idx = []
    for net_i, dataidx in net_dataidx_map.items():
        if net_i in supervised_user_id:
            net_client_sup_idx.extend(dataidx)
        net_client_all_idx.extend(dataidx)

    online_classifier_glob: nn.Module = nn.Linear(net_glob.backbone.fc.in_features, n_classes)
    online_classifier_glob.train()

    cls_glob = online_classifier_glob.state_dict()

    _, _, _, _, _, test_class_centeroids, test_class_count = test(0,
                                                                  net_glob.backbone.state_dict(),
                                                                  online_classifier_glob.state_dict(),
                                                                  X_test, y_test,
                                                                  n_classes)
    # test_acc_1 = knn_test(args, net_glob.backbone, X_train[net_client_all_idx], y_train[net_client_all_idx],
    #                       X_test, y_test, n_classes, device='cuda')
    # logger.info(
    #     "\n K-NN TEST Acc: {:6f}"
    #     .format(test_acc_1))
    # pseudo labelling and training
    T_base = 0.84

    # T_lower = args.b / n_classes

    T_higher = 0.1
    T_upper = 0.95
    all_local = []
    # load number of classes in labeled clients

    # sup_label = torch.load('partition_strategy/svhn_beta0.8_sup.pth')
    sup_label = [0] * n_classes
    for i in range(n_classes):
        sup_label[i] = sum([sum(y_train[net_dataidx_map[c_i]] == i) for c_i in supervised_user_id])

    logger.info('cumulative supervised labels'
                )
    logger.info(sup_label)
    if args.fed_flex:
        class_confident = torch.tensor(copy.deepcopy(sup_label / max(sup_label))).cuda()
    else:
        temp_sup_label = copy.deepcopy(sup_label)

        temp_sup_label = (temp_sup_label / sum(temp_sup_label)) * (n_classes / 10)

        temp_sup_label = temp_sup_label

        class_confident = temp_sup_label + T_base - temp_sup_label.std()

        if args.dataset == 'skin' or args.dataset == 'SVHN':
            class_confident[class_confident >= 0.9] = 0.9
        else:
            class_confident[class_confident >= T_upper] = T_upper

    print(class_confident)

    if args.warmup:
        args.rounds = args.warmup_comm_round
        for i in supervised_user_id + unsupervised_user_id:
            lab_trainer_locals.append(SelfsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
            w_locals.append(copy.deepcopy(w_glob))
            sup_net_locals.append(copy.deepcopy(net_glob))
            sup_classifier_locals.append(copy.deepcopy(online_classifier_glob))

            if args.opt == 'adam':
                optimizer = torch.optim.Adam(sup_net_locals[i].parameters(), lr=args.base_lr,
                                             betas=(0.9, 0.999), weight_decay=5e-4)
            elif args.opt == 'sgd':
                learnable_params = [
                    {"name": "backbone", "params": sup_net_locals[i].parameters()},
                    {
                        "name": "classifier",
                        "params": sup_classifier_locals[i].parameters(),
                        "lr": 0.1,
                        "weight_decay": 0,
                    },
                ]
                optimizer = torch.optim.SGD(learnable_params, lr=args.base_lr, momentum=0.9,
                                            weight_decay=5e-4)

                if args.timm_cos:
                    scheduler = CosineLRScheduler(optimizer, t_initial=args.rounds * args.local_ep,
                                                  warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)

            elif args.opt == 'adamw':
                optimizer = torch.optim.AdamW(sup_net_locals[i].parameters(), lr=args.base_lr, weight_decay=0.02)
            elif args.opt == 'lars':
                scale_factor = args.batch_size / 256
                learnable_params = [
                    {"name": "backbone", "params": sup_net_locals[i].parameters()},
                    {
                        "name": "classifier",
                        "params": sup_classifier_locals[i].parameters(),
                        "lr": 0.1,
                        "weight_decay": 0,
                    },
                ]
                learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)
                idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]

                optimizer = LARS(learnable_params, lr=args.base_lr * scale_factor, weight_decay=1e-4,
                                 clip_lr=True,
                                 eta=0.02,
                                 exclude_bias_n_norm=True, momentum=0.9)
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=10,
                    max_epochs=args.rounds * args.local_ep,
                    warmup_start_lr=0.00003,
                    eta_min=0,
                )

                if idxs_no_scheduler:
                    partial_fn = partial(
                        static_lr,
                        get_lr=scheduler.get_lr,
                        param_group_indexes=idxs_no_scheduler,
                        lrs_to_replace=[args.base_lr * scale_factor] * len(idxs_no_scheduler),
                    )
                    scheduler.get_lr = partial_fn
            if args.opt == 'lars' or args.timm_cos:
                sup_optim_locals.append({"optimizer": copy.deepcopy(optimizer.state_dict()),
                                         "scheduler": copy.deepcopy(scheduler.state_dict())})
            else:
                sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

        criterion_client_dict = {}

        # supervised training in labeled clients, change com_round if number of labeled clients > 1
        for com_round in trange(args.warmup_comm_round):
            print("************* Communication round %d begins *************" % com_round)
            w_l = []
            cls_l = []
            local_num = []
            loss_locals = []
            for client_idx in supervised_user_id + unsupervised_user_id:
                local = lab_trainer_locals[client_idx]
                optimizer = sup_optim_locals[client_idx]
                train_dl_local, train_ds_local = get_dataloader(args,
                                                                X_train[net_dataidx_map[client_idx]],
                                                                y_train[net_dataidx_map[client_idx]],
                                                                args.dataset,
                                                                args.datadir, args.batch_size, is_labeled=False,
                                                                data_idxs=net_dataidx_map[client_idx],
                                                                pre_sz=args.pre_sz, input_sz=args.input_sz)

                if criterion_client_dict.get(client_idx, 0) == 0:
                    class_count_per_client = [0] * (n_classes)
                    class_idx, class_count_per_client_tr = np.unique(train_ds_local.labels,
                                                                     return_counts=True)
                    for idx, j in enumerate(class_idx):
                        class_count_per_client[j] = class_count_per_client_tr[idx]
                    loss_fn = get_criterion(class_count_per_client, args.warmup_comm_round, args.loss_fn_name)
                    criterion_client_dict[client_idx] = loss_fn

                w, cls_w, loss, loss_cl, op, class_centeroids, class_count, num, train_label, class_cos_ssl, pl_bacc, class_cos_ssl_all = local.train(
                    args,
                    sup_net_locals[client_idx].state_dict(),
                    sup_classifier_locals[client_idx].state_dict(),
                    optimizer,
                    com_round * args.local_ep,
                    client_idx,
                    train_dl_local, n_classes, loss_fn=criterion_client_dict[client_idx])

                if args.vis_collapse:
                    for class_idx_, cos_value_class in enumerate(class_cos_ssl_all.squeeze().cpu().numpy().tolist()):
                        wandb.log(
                            {'com_round': com_round, f'distance_cos_l{class_idx_}_c{client_idx}': cos_value_class},
                            step=com_round)

                if args.vis_ph:
                    cos_sim = torch.cosine_similarity(class_centeroids[class_count != 0],
                                                      test_class_centeroids[class_count != 0])
                    wandb.log({'com_round': com_round, f'self-sup_test_cos_{client_idx}': torch.mean(cos_sim).item()},
                              step=com_round)
                if args.opt == 'lars' or args.timm_cos:
                    lr_ = sup_optim_locals[
                        client_idx]['optimizer'][
                        'param_groups'][0][
                        'lr']
                else:
                    lr_ = sup_optim_locals[
                        client_idx][
                        'param_groups'][0][
                        'lr']
                # writer.add_scalar('Supervised loss on sup client %d' % client_idx, loss, global_step=com_round)
                wandb.log({'com_round': com_round, f'self-sup_train_loss_{client_idx}': loss, 'lr': lr_},
                          step=com_round)

                w_l.append(copy.deepcopy(w))
                cls_l.append(copy.deepcopy(cls_w))

                sup_optim_locals[client_idx] = copy.deepcopy(op)
                loss_locals.append(copy.deepcopy(loss))
                logger.info(
                    'Self-supervised client {} sample num: {} training loss : {} lr : {}'.format(client_idx,
                                                                                                 len(train_ds_local),
                                                                                                 loss,
                                                                                                 lr_))
                local_num.append(len(net_dataidx_map[client_idx]))

            total_lenth_this = sum(local_num)
            clt_freq_this_round = [i / total_lenth_this for i in local_num]

            w_glob = FedAvg(w_l, clt_freq_this_round)
            cls_glob = FedAvg(cls_l, clt_freq_this_round)

            net_glob.load_state_dict(w_glob)
            online_classifier_glob.load_state_dict(cls_glob)

            # if com_round % 10 == 0:
            # test_acc_1 = knn_test(args, net_glob.backbone, X_train[net_client_all_idx], y_train[net_client_all_idx],
            #                       X_test, y_test, n_classes, device='cuda')

            # wandb.log({'com_round': com_round,
            #            'K-NN-Acc': test_acc_1,
            #            }, step=com_round)

            # logger.info("\nK-NN TEST Student: Epoch: {}".format(com_round))
            # logger.info(
            #     "\n K-NN TEST Acc: {:6f}"
            #     .format(test_acc_1))
            AUROC_avg, Accus_avg, Pre, Recall, bAccus_avg, test_class_centeroids, test_class_count = test(com_round,
                                                                                                          net_glob.backbone.state_dict(),
                                                                                                          online_classifier_glob.state_dict(),
                                                                                                          X_test,
                                                                                                          y_test,
                                                                                                          n_classes)

            wandb.log({'com_round': com_round,
                       'AUC': AUROC_avg,
                       'Acc': Accus_avg,
                       'B-Acc': bAccus_avg,
                       'Pre': Pre,
                       'Recall': Recall,
                       }, step=com_round)

            logger.info("\nTEST Student: Epoch: {}".format(com_round))
            logger.info("\nTEST AUROC: {:6f}, TEST B-Acc: {:6f}, TEST Accus: {:6f}, TEST Pre: {:6f}, TEST Recall: {:6f}"
                        .format(AUROC_avg, bAccus_avg, Accus_avg, Pre, Recall))

            if com_round % 100 == 0:
                logreg = torch.nn.Sequential(torch.nn.Linear(net_glob.backbone.fc.in_features, n_classes))
                logreg = logreg.cuda()
                net_glob.eval()

                AUROC_avg, Accus_avg, Pre, Recall, bAccus_avg = test_fed_ssl(args, net_glob.backbone, logreg,
                                                                             X_train[net_client_all_idx],
                                                                             y_train[net_client_all_idx], X_test,
                                                                             y_test, num_classes=n_classes,
                                                                             )

                wandb.log({'com_round': com_round,
                           'Upper-AUC': AUROC_avg,
                           'Upper-Acc': Accus_avg,
                           'Upper-B-Acc': bAccus_avg,
                           'Upper-Pre': Pre,
                           'Upper-Recall': Recall,
                           }, step=com_round)

                # self.model.cpu()
                logreg = torch.nn.Sequential(torch.nn.Linear(net_glob.backbone.fc.in_features, n_classes))
                logreg = logreg.cuda()

                AUROC_avg, Accus_avg, Pre, Recall, bAccus_avg = test_fed_ssl(args, net_glob.backbone, logreg,
                                                                             X_train[net_client_sup_idx],
                                                                             y_train[net_client_sup_idx], X_test,
                                                                             y_test, num_classes=n_classes,
                                                                             )

                wandb.log({'com_round': com_round,
                           'Lower-AUC': AUROC_avg,
                           'Lower-Acc': Accus_avg,
                           'Lower-B-Acc': bAccus_avg,
                           'Lower-Pre': Pre,
                           'Lower-Recall': Recall,
                           }, step=com_round)

                net_glob.train()

            for i in supervised_user_id + unsupervised_user_id:
                sup_net_locals[i].load_state_dict(w_glob)
                sup_classifier_locals[i].load_state_dict(cls_glob)

            if com_round == args.rounds // 2 and not args.vis_collapse:
                torch.save(net_glob.state_dict(),
                           f'warmup_ssl/{args.dataset}_res_{args.scale_loss}_{com_round}_timcos_{args.timm_cos}_{len(supervised_user_id)}_{len(unsupervised_user_id)}_{args.loss_fn_name}_{args.model}_{args.ssl_model}opt_{args.opt}_lr_{args.base_lr}_{args.warmup_comm_round}_beta0.8.pth')

        net_glob.load_state_dict(w_glob)
        online_classifier_glob.load_state_dict(cls_glob)
        if not args.vis_collapse:
            torch.save(net_glob.state_dict(),
                       f'warmup_ssl/{args.dataset}_res_{args.scale_loss}_{com_round}_timcos_{args.timm_cos}_{len(supervised_user_id)}_{len(unsupervised_user_id)}_{args.loss_fn_name}_{args.model}_{args.ssl_model}opt_{args.opt}_lr_{args.base_lr}_{args.warmup_comm_round}_beta0.8.pth')

        AUROC_avg, Accus_avg, Pre, Recall, bAccus_avg, test_class_centeroids, test_class_count = test(com_round,
                                                                                                      net_glob.backbone.state_dict(),
                                                                                                      online_classifier_glob.state_dict(),
                                                                                                      X_test, y_test,
                                                                                                      n_classes)

        wandb.log({'com_round': com_round,
                   'AUC': AUROC_avg,
                   'Acc': Accus_avg,
                   'B-Acc': bAccus_avg,
                   'Pre': Pre,
                   'Recall': Recall,
                   }, step=com_round)

        logger.info("\nTEST Student: Epoch: {}".format(com_round))
        logger.info("\nTEST AUROC: {:6f}, TEST B-Acc: {:6f}, TEST Accus: {:6f}, TEST Pre: {:6f}, TEST Recall: {:6f}"
                    .format(AUROC_avg, bAccus_avg, Accus_avg, Pre, Recall))

        logreg = torch.nn.Sequential(torch.nn.Linear(net_glob.backbone.fc.in_features, n_classes))
        logreg = logreg.cuda()
        net_glob.eval()

        AUROC_avg, Accus_avg, Pre, Recall, bAccus_avg = test_fed_ssl(args, net_glob.backbone, logreg,
                                                                     X_train[net_client_all_idx],
                                                                     y_train[net_client_all_idx], X_test, y_test,
                                                                     num_classes=n_classes,
                                                                     )

        wandb.log({'com_round': com_round,
                   'Upper-AUC': AUROC_avg,
                   'Upper-Acc': Accus_avg,
                   'Upper-B-Acc': bAccus_avg,
                   'Upper-Pre': Pre,
                   'Upper-Recall': Recall,
                   }, step=com_round)

        # self.model.cpu()
        logreg = torch.nn.Sequential(torch.nn.Linear(net_glob.backbone.fc.in_features, n_classes))
        logreg = logreg.cuda()

        AUROC_avg, Accus_avg, Pre, Recall, bAccus_avg = test_fed_ssl(args, net_glob.backbone, logreg,
                                                                     X_train[net_client_sup_idx],
                                                                     y_train[net_client_sup_idx], X_test, y_test,
                                                                     num_classes=n_classes,
                                                                     )

        wandb.log({'com_round': com_round,
                   'Lower-AUC': AUROC_avg,
                   'Lower-Acc': Accus_avg,
                   'Lower-B-Acc': bAccus_avg,
                   'Lower-Pre': Pre,
                   'Lower-Recall': Recall,
                   }, step=com_round)

        exit()

    else:
        ##resuming after warmup
        if args.resume:
            print('==> Resuming from checkpoint..')
            if args.ssl_pretrain:
                if args.dataset == 'cifar100':
                    if args.long_tailed:
                        ssl_checkpoint = 'cifar100_LT_res_0.1_500_timcos_True_1_9_BSM_Res18_cifar_barlowopt_sgd_lr_0.008_1000_beta0.8.pth'
                        clr_checkpoints = 'cifar100_LT_res_1_9_BSM_Res18_cifar_opt_sgd_lr_0.1_256_beta0.8.pth'
                    else:
                        ssl_checkpoint = 'cifar100_res_500_timcos_True_1_9_CE_Res18_cifar_barlowopt_sgd_lr_0.008_1000_beta0.8.pth'
                        clr_checkpoints = 'cifar100_res_1_9_BSM_Res18_cifar_opt_sgd_lr_0.1_256_beta0.8.pth'
                elif args.dataset == 'SVHN':
                    ssl_checkpoint = 'SVHN_res_250_timcos_True_1_9_CE_Res18_cifar_barlowopt_sgd_lr_0.016_500_beta0.8.pth'
                    clr_checkpoints = 'SVHN_res_1_9_BSM_Res18_cifar_opt_sgd_lr_0.1_256_beta0.8.pth'
                elif args.dataset == 'skin':
                    ssl_checkpoint = 'skin_res_1_9_BSM_Res18_opt_sgd_lr_0.016_500_beta0.8.pth'
                    clr_checkpoints = 'skin_res_1_9_BSM_Res18_opt_sgd_lr_0.1_256_beta0.8.pth'

                checkpoint = torch.load(
                    f'warmup_ssl/{ssl_checkpoint}'
                )

                clr_checkpoint = torch.load(
                    f'warmup_clr/{clr_checkpoints}')

                net_glob.load_state_dict(checkpoint)
                online_classifier_glob.load_state_dict(clr_checkpoint)
                net_glob.backbone.fc.load_state_dict(online_classifier_glob.state_dict())

            else:
                if args.pl:
                    warmup_directory_name = 'warmup_pl'
                else:
                    if args.num_users != 10 or args.beta != 0.8:
                        warmup_directory_name = f'warmup_{args.num_users}_{args.beta}'

                    else:
                        warmup_directory_name = 'warmup'

                if args.long_tailed:
                    checkpoint = torch.load(
                        f'{warmup_directory_name}/{args.dataset}_LT_res_{len(supervised_user_id)}_{args.loss_fn_name}_{args.model}_opt_{args.opt}_lr_{args.base_lr}_beta0.8.pth')
                else:
                    checkpoint = torch.load(
                        f'{warmup_directory_name}/{args.dataset}_res_{len(supervised_user_id)}_{args.loss_fn_name}_{args.model}_opt_{args.opt}_lr_{args.base_lr}_beta0.8.pth')

                net_glob.backbone.load_state_dict(checkpoint)

                online_classifier_glob.load_state_dict(net_glob.backbone.fc.state_dict())

            start_epoch = 0
        else:
            start_epoch = 0

    for i in supervised_user_id + unsupervised_user_id:
        if i in supervised_user_id:
            if args.pl:
                lab_trainer_locals.append(PLFlexSelfsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes
                                                                          , add_cls_to_optim=True))
            elif args.remove_n:
                lab_trainer_locals[i] = SelfsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes
                                                                  , add_cls_to_optim=True)
            else:
                lab_trainer_locals.append(SelfsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes
                                                                    , add_cls_to_optim=True))
        else:
            if args.flex:
                if args.remove_n:
                    lab_trainer_locals[i] = FlexSelfsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes
                                                                          , add_cls_to_optim=True)
                else:
                    lab_trainer_locals.append(FlexSelfsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes
                                                                            , add_cls_to_optim=True))
            elif args.fix:
                lab_trainer_locals.append(FixSelfsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes
                                                                       , add_cls_to_optim=True))
            else:
                lab_trainer_locals.append(PLSelfsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes
                                                                      , add_cls_to_optim=True))

        w_locals.append(copy.deepcopy(w_glob))
        if args.remove_n:
            sup_net_locals[i] = copy.deepcopy(net_glob)
            sup_classifier_locals[i] = copy.deepcopy(online_classifier_glob)
        else:
            sup_net_locals.append(copy.deepcopy(net_glob))

            sup_classifier_locals.append(copy.deepcopy(online_classifier_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(sup_net_locals[i].parameters(), lr=args.base_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            if i in supervised_user_id or i in unsupervised_user_id:
                logger.info('classifier into supervised client')
                learnable_params = [
                    {"name": "backbone", "params": sup_net_locals[i].parameters()},
                    {
                        "name": "classifier",
                        "params": sup_classifier_locals[i].parameters(),
                        "lr": args.base_lr if i in supervised_user_id else args.unsup_lr,
                        "weight_decay": 0,
                    },
                ]
            else:
                learnable_params = [
                    {"name": "backbone", "params": sup_net_locals[i].parameters()},
                ]

            optimizer = torch.optim.SGD(learnable_params, lr=args.base_lr if i in supervised_user_id else args.unsup_lr,
                                        momentum=0.9,
                                        weight_decay=5e-4)
            if args.timm_cos:
                scheduler = CosineLRScheduler(optimizer, t_initial=args.rounds * args.local_ep,
                                              warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6, cycle_decay=0.1)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(sup_net_locals[i].parameters(), lr=args.base_lr, weight_decay=0.02)

        if args.opt == 'lars' or args.timm_cos:
            sup_optim_locals.append({"optimizer": copy.deepcopy(optimizer.state_dict()),
                                     "scheduler": copy.deepcopy(scheduler.state_dict())})
        else:
            if args.remove_n:
                sup_optim_locals[i] = copy.deepcopy(optimizer.state_dict())
            else:
                sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    criterion_client_dict = {}
    train_dl_local_unl = None
    # supervised training in labeled clients, change com_round if number of labeled clients > 1
    for com_round in trange(start_epoch, args.rounds):
        print("************* Comm round %d begins *************" % com_round)
        w_l = []
        cls_l = []
        local_num = []
        local_num_pl = []
        loss_locals = []
        local_label = torch.zeros(n_classes)
        class_centeroids_all = torch.zeros((total_num, n_classes)).cuda()
        bacc_list = []
        logger.info(f'Comm round {com_round}')
        for client_idx in supervised_user_id + unsupervised_user_id:
            local = lab_trainer_locals[client_idx]
            optimizer = sup_optim_locals[client_idx]
            train_dl_local, train_ds_local = get_dataloader(args,
                                                            X_train[net_dataidx_map[client_idx]],
                                                            y_train[net_dataidx_map[client_idx]],
                                                            args.dataset,
                                                            args.datadir, args.batch_size, is_labeled=False,
                                                            data_idxs=net_dataidx_map[client_idx],
                                                            pre_sz=args.pre_sz, input_sz=args.input_sz)
            if criterion_client_dict.get(client_idx, 0) == 0:
                class_count_per_client = [0] * (n_classes)
                class_idx, class_count_per_client_tr = np.unique(train_ds_local.labels,
                                                                 return_counts=True)
                for idx, j in enumerate(class_idx):
                    class_count_per_client[j] = class_count_per_client_tr[idx]
                loss_fn = get_criterion(class_count_per_client, args.rounds, args.loss_fn_name)
                criterion_client_dict[client_idx] = loss_fn
            if args.pl:
                train_dl_local_unl, train_ds_local_unl = get_dataloader(args,
                                                                        X_train[net_dataidx_map_un[client_idx]],
                                                                        y_train[net_dataidx_map_un[client_idx]],
                                                                        args.dataset,
                                                                        args.datadir, args.batch_size, is_labeled=False,
                                                                        data_idxs=net_dataidx_map_un[client_idx],
                                                                        pre_sz=args.pre_sz, input_sz=args.input_sz)

            w, cls_w, loss, loss_cl, op, class_centeroids, class_count, num, train_label, class_cos_ssl, pl_bacc, class_cos_ssl_all = local.train(
                args,
                sup_net_locals[client_idx].state_dict(),
                sup_classifier_locals[client_idx].state_dict(),
                optimizer,
                com_round * args.local_ep,
                client_idx,
                train_dl_local, n_classes, loss_fn=criterion_client_dict[client_idx],
                train_classifier=True, fix_match=False, class_confident=class_confident, client_idx=client_idx,
                train_dl_local_unl=train_dl_local_unl)

            bacc_list.append(pl_bacc)
            class_centeroids_all[client_idx, :] = class_cos_ssl.squeeze()

            if client_idx in unsupervised_user_id:
                local_label = local_label + train_label
                local_num.append(num)
            else:
                local_num.append(len(net_dataidx_map[client_idx]))
                if args.pl:
                    local_label = local_label + train_label
                    local_num_pl.append(num)

            if args.vis_collapse:
                for class_idx_, cos_value_class in enumerate(class_cos_ssl_all.squeeze().cpu().numpy().tolist()):
                    wandb.log({'com_round': com_round, f'distance_cos_l{class_idx_}_c{client_idx}': cos_value_class},
                              step=com_round)

            if args.vis_ph:
                cos_sim = torch.cosine_similarity(class_centeroids[class_count != 0],
                                                  test_class_centeroids[class_count != 0])
                wandb.log({'com_round': com_round, f'self-sup_test_cos_{client_idx}': torch.mean(cos_sim).item()},
                          step=com_round)
            if args.opt == 'lars' or args.timm_cos:
                lr_ = sup_optim_locals[
                    client_idx]['optimizer'][
                    'param_groups'][0][
                    'lr']
            else:
                lr_ = sup_optim_locals[
                    client_idx][
                    'param_groups'][0][
                    'lr']
            # writer.add_scalar('Supervised loss on sup client %d' % client_idx, loss, global_step=com_round)
            wandb.log(
                {'com_round': com_round, f'self-sup_train_loss_{client_idx}': loss,
                 f'self-sup_train_cls_loss_{client_idx}': loss_cl,
                 'lr': lr_},
                step=com_round)

            w_l.append(copy.deepcopy(w))
            cls_l.append(copy.deepcopy(cls_w))

            sup_optim_locals[client_idx] = copy.deepcopy(op)
            loss_locals.append(copy.deepcopy(loss))
            logger.info(
                'Self-supervised client {} sample num: {} training loss : {}, clr loss: {} lr : {}'.format(client_idx,
                                                                                                           len(train_ds_local),
                                                                                                           loss,
                                                                                                           loss_cl,
                                                                                                           lr_))
        print(local_num)
        local_label = local_label + torch.Tensor(sup_label)

        if args.fed_flex:
            if args.thresh_warmup:
                local_label = local_label / max(local_label)
            else:
                local_label = local_label / max(each_lenth)

            class_confident = copy.deepcopy(local_label).cuda()
        else:
            local_label = (local_label / sum(local_label)) * (n_classes / 10)

            class_confident = local_label + T_base - local_label.std()

            if args.dataset == 'skin' or args.dataset == 'SVHN':
                class_confident[class_confident >= 0.9] = 0.9
            else:
                class_confident[class_confident >= T_upper] = T_upper

        print(class_confident * args.main_T)

        total_lenth_this = sum(local_num[:len(supervised_user_id)])
        if args.remove_n:
            clt_freq_this_round = [i / total_lenth_this for i in local_num[:len(supervised_user_id)]]
            cls_glob_supervised = FedAvg([cls_l[i] for i in range(len(supervised_user_id))], clt_freq_this_round)
        else:
            clt_freq_this_round = [i / total_lenth_this for i in local_num[:len(supervised_user_id)]]
            cls_glob_supervised = FedAvg([cls_l[i] for i in supervised_user_id], clt_freq_this_round)

        w_glob_supervised = FedAvg([w_l[i] for i in supervised_user_id], clt_freq_this_round)

        if args.pl:
            total_lenth_this = sum(local_num_pl[:len(supervised_user_id)])
        else:
            total_lenth_this = sum(local_num[len(supervised_user_id):])

        if total_lenth_this == 0:
            w_glob_unsupervised = copy.deepcopy(w_glob_supervised)
            cls_glob_unsupervised = copy.deepcopy(cls_glob_supervised)
        else:
            if args.pl:
                class_centeroids_all = class_centeroids_all[:len(supervised_user_id)].squeeze()
            elif args.remove_n:
                indices = torch.arange(class_centeroids_all.size(0)) != args.remove_n
                class_centeroids_all = class_centeroids_all[indices]
                class_centeroids_all = class_centeroids_all[len(supervised_user_id):].squeeze()
            else:
                class_centeroids_all = class_centeroids_all[len(supervised_user_id):].squeeze()

            class_centeroids_per_class = class_centeroids_all / torch.sum(class_centeroids_all, dim=0)
            value_to_aggregate = torch.sum((1 - class_centeroids_per_class), dim=1)

            clt_freq_this_round = value_to_aggregate / torch.sum(value_to_aggregate)
            clt_freq_this_round = clt_freq_this_round.tolist()

            if args.pl:
                list_mas_to_follow = supervised_user_id
            else:
                list_mas_to_follow = unsupervised_user_id

            if args.remove_n:
                list_mas_to_follow_agg = range(len(supervised_user_id), len(unsupervised_user_id))
            else:
                list_mas_to_follow_agg = list_mas_to_follow

            w_glob_unsupervised = FedAvg([w_l[i] for i in list_mas_to_follow_agg], clt_freq_this_round)
            cls_glob_unsupervised = FedAvg([cls_l[i] for i in list_mas_to_follow_agg], clt_freq_this_round)

            for i in list_mas_to_follow_agg:
                wandb.log(
                    {'com_round': com_round,
                     f'mas_{i}': clt_freq_this_round[i] if args.pl else clt_freq_this_round[
                         args.num_labeled - i],
                     f'psuedo-bacc_{i}': bacc_list[i],
                     },
                    step=com_round)


        w_glob = FedAvg([w_glob_supervised, w_glob_unsupervised], [0.5, 0.5])
        cls_glob = FedAvg([cls_glob_supervised, cls_glob_unsupervised], [0.5, 0.5])


        net_glob.load_state_dict(w_glob)
        online_classifier_glob.load_state_dict(cls_glob)

        AUROC_avg, Accus_avg, Pre, Recall, bAccus_avg, test_class_centeroids, test_class_count = test(com_round,
                                                                                                      net_glob.backbone.state_dict(),
                                                                                                      online_classifier_glob.state_dict(),
                                                                                                      X_test,
                                                                                                      y_test,
                                                                                                      n_classes)

        wandb.log({'com_round': com_round,
                   'AUC': AUROC_avg,
                   'Acc': Accus_avg,
                   'B-Acc': bAccus_avg,
                   'Pre': Pre,
                   'Recall': Recall,
                   }, step=com_round)

        logger.info("\nTEST Student: Epoch: {}".format(com_round))
        logger.info("\nTEST AUROC: {:6f}, TEST B-Acc: {:6f}, TEST Accus: {:6f}, TEST Pre: {:6f}, TEST Recall: {:6f}"
                    .format(AUROC_avg, bAccus_avg, Accus_avg, Pre, Recall))

        AUROC_avg, Accus_avg, Pre, Recall, bAccus_avg, _, _ = test(com_round,
                                                                   net_glob.backbone.state_dict(),
                                                                   cls_glob_supervised,
                                                                   X_test, y_test,
                                                                   n_classes)
        wandb.log({'com_round': com_round,
                   'sup-clr-AUC': AUROC_avg,
                   'sup-clr-Acc': Accus_avg,
                   'sup-clr-B-Acc': bAccus_avg,
                   'sup-clr-Pre': Pre,
                   'sup-clr-Recall': Recall,
                   }, step=com_round)

        logger.info("\nsup-clr- Student: Epoch: {}".format(com_round))
        logger.info("\nsup-clr- AUROC: {:6f}, TEST B-Acc: {:6f}, TEST Accus: {:6f}, TEST Pre: {:6f}, TEST Recall: {:6f}"
                    .format(AUROC_avg, bAccus_avg, Accus_avg, Pre, Recall))

        online_classifier_glob.train()
        net_glob.train()

        for i in supervised_user_id + unsupervised_user_id:
            sup_net_locals[i].load_state_dict(w_glob)
            sup_classifier_locals[i].load_state_dict(cls_glob)

    if args.unsup_num == 0 and args.num_labeled == 1 and args.ssl_pretrain:
        torch.save(net_glob.state_dict(),
                   f'warmup_ssl_again/{args.dataset}_res_{com_round}_timcos_{args.timm_cos}_{len(supervised_user_id)}_{len(unsupervised_user_id)}_{args.loss_fn_name}_{args.model}_{args.ssl_model}opt_{args.opt}_lr_{args.base_lr}_{args.warmup_comm_round}_beta0.8.pth')

        torch.save(online_classifier_glob.state_dict(),
                   f'warmup_ssl_again_clr/{args.dataset}_res_{com_round}_timcos_{args.timm_cos}_{len(supervised_user_id)}_{len(unsupervised_user_id)}_{args.loss_fn_name}_{args.model}_{args.ssl_model}opt_{args.opt}_lr_{args.base_lr}_{args.warmup_comm_round}_beta0.8.pth')

    try:
        pass
    except Exception as e:
        wandb.finish()
