import os
import pickle
import torch
from cifar_load import partition_data_allnoniid, get_dataloader
from networks.ssl_models import get_model
from options import args_parser
import numpy as np
import torch.nn as nn
from utils.utils_SimPLE import get_class_compose

# to init cuda before importing torch
args = args_parser()

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

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(model, train_loader, test_loader, device,psuedo=False):
    train_X, train_y = inference(train_loader, model, device,psuedo)
    test_X, test_y = inference(test_loader, model, device,psuedo)
    return train_X, train_y, test_X, test_y

def NormalizeTensor(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


if __name__ == '__main__':
    print(dict(vars(args)))
    train_idxs = None
    test_idxs = None

    if args.dataset == 'SVHN':
        partition = torch.load('partition_strategy/SVHN_noniid_10%labeled.pth')
        net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar100':
        if args.long_tailed:
            with open('partition_strategy/cifar100_LT_noniid_10%labeled.pth', 'rb') as f:  # Python 3: open(..., 'rb')
                net_dataidx_map, LT_idx = pickle.load(f)
        else:
            partition = torch.load('partition_strategy/cifar100_noniid_10%labeled.pth')
            net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar10':
        partition = torch.load('partition_strategy/cifar10_0.8.pth')
        net_dataidx_map = partition

    elif args.dataset == 'skin':
        partition = torch.load('partition_strategy/skin_noniid_beta0.8.pth')

        # partition = torch.load('partition_strategy/skin_noniid_beta0.8.pth')
        net_dataidx_map = partition['data_partition']
        train_idxs = partition['train_list']
        test_idxs = partition['test_list']

    # because only load_skin needs train and test ids

    X_train, y_train, X_test, y_test, _, traindata_cls_counts = partition_data_allnoniid(
        args.dataset, args.datadir, train_idxs=train_idxs, test_idxs=test_idxs, partition=args.partition,
        n_parties=args.num_users, beta=args.beta)

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

    net_glob_ssl = get_model(args, n_classes)



    start_epoch = 0
    net_glob_ssl.eval()
    net_glob_ssl.cuda()

    y_train_psuedo = np.zeros_like(y_train)
    client_n = args.num_users
    class_centeroids_all = torch.zeros((client_n, n_classes)).cuda()
    class_prior_all = torch.zeros((client_n, n_classes)).cuda()
    psuedo_dist_all = torch.zeros((client_n, n_classes)).cuda()
    bacc_list=[]
    for client_idx in range(0, client_n):
        class_centeroids = torch.zeros(n_classes, 1).cuda()

        class_centeroid_ssl = torch.zeros(n_classes, 1).cuda()

        train_dl_local, train_ds_local = get_dataloader(args,
                                                        X_train[net_dataidx_map[client_idx]],
                                                        y_train[net_dataidx_map[client_idx]],
                                                        args.dataset,
                                                        args.datadir, args.batch_size, is_labeled=True, is_testing=True,
                                                        data_idxs=net_dataidx_map[client_idx],
                                                        pre_sz=args.pre_sz, input_sz=args.input_sz)

        class_count_per_client = [0] * (n_classes)
        class_idx, class_count_per_client_tr = np.unique(train_ds_local.labels,
                                                         return_counts=True)
        for idx, j in enumerate(class_idx):
            class_count_per_client[j] = class_count_per_client_tr[idx]

        class_count_per_client = np.array(class_count_per_client) / sum(class_count_per_client)

        class_count = torch.tensor(np.array(get_class_compose(train_dl_local, n_classes))).cuda()


        with torch.no_grad():
            (train_X_ssl, _, _, _) = get_features(
                net_glob_ssl.backbone, train_dl_local, [], device='cuda'
            )


        os.makedirs(f'data_rand/{args.dataset}',exist_ok=True)

        if args.long_tailed:
            torch.save(nn.functional.normalize(torch.from_numpy(train_X_ssl),
                                               dim=-1),
                       f'data_rand/{args.dataset}_LT/{client_idx}.pt')

        else:
            torch.save(nn.functional.normalize(torch.from_numpy(train_X_ssl),
                                               dim=-1),
                       f'data_rand/{args.dataset}/{client_idx}.pt')



