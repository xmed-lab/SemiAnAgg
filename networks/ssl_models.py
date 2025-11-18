import copy

import torch
from torch import nn

from networks.orchestra import simclr, simsiam, byol, byol_free, specloss, rotpred, orchestra
from networks.resnet import resnet18
from networks.resnetcifar import ResNet18_cifar10

import torch.nn.functional as F

TwoLayer = "2_layer"
OneLayer = "1_layer"


def get_model(args, n_classes):
    if args.model == 'Res18':
        backbone = resnet18(n_classes, pretrained=args.Pretrained, KD=True)
    elif args.model == 'Res18_cifar':
        backbone = ResNet18_cifar10(num_classes=n_classes, KD=True)

    if args.ssl_model == 'barlow':
        return BarlowTwins(args, backbone)
    elif args.ssl_model == 'byol':
        stop_gradient = True
        has_predictor = True
        return BYOLModel(net=backbone, stop_gradient=stop_gradient, has_predictor=has_predictor,
                         predictor_network=TwoLayer)

    else:
        raise NotImplementedError

def get_model_ssl(args,n_classes):
    config_dict=vars(args)
    config_dict['n_classes']=n_classes
    if (args.ssl_model == "simclr"):
        net = simclr(config_dict=config_dict, bbone_arch=args.model)
    elif (args.ssl_model == "simsiam"):
        net = simsiam(config_dict=config_dict, bbone_arch=args.model)
    elif (args.ssl_model == "byol"):
        net = byol(config_dict=config_dict, bbone_arch=args.model)
    elif (args.ssl_model == "byol_free"):
        net = byol_free(config_dict=config_dict, bbone_arch=args.model)
    elif (args.ssl_model == "specloss"):
        net = specloss(config_dict=config_dict, bbone_arch=args.model)
    elif (args.ssl_model == "rotpred"):
        net = rotpred(config_dict=config_dict, bbone_arch=args.model)
    elif (args.ssl_model == "orchestra"):
        net = orchestra(config_dict=config_dict, bbone_arch=args.model)

    return net
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args

        self.backbone = backbone
        # self.backbone.fc = nn.Identity()
        self.feat_num = backbone.fc.in_features
        # 8192 - 8192 - 8192
        # projector
        sizes = [self.feat_num] + [self.feat_num * 4, self.feat_num * 4, self.feat_num * 4]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def barlow_loss_fn(self, z1, z2, batch_size, lambd):
        # empirical cross-correlation matrix
        c = z1.T @ z2
        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        return loss

    def forward(self, y1, y2=None, train_method='ssl'):
        if train_method == 'ssl':
            p1, _ = self.backbone(y1)
            p2, _ = self.backbone(y2)
            # print(p1.size())
            # if len(p1) != 3:
            #     p1 = p1.unsqueeze(dim=0)
            #     p2 = p2.unsqueeze(dim=0)

            z1 = self.projector(p1)
            z2 = self.projector(p2)
            z1, z2 = self.bn(z1), self.bn(z2)

            loss_u = self.barlow_loss_fn(z1, z2, self.args.batch_size, self.args.lambd)

            return p1, p2, loss_u
        elif train_method == 'ssl':
            pass
        else:
            p1, _ = self.backbone(y1)
            return


class BYOLModel(nn.Module):
    def __init__(
            self,
            net=None,
            projection_size=2048,
            projection_hidden_size=4096,
            moving_average_decay=0.99,
            stop_gradient=True,
            has_predictor=True,
            predictor_network=TwoLayer,
    ):
        super().__init__()

        self.backbone = net
        if not hasattr(net, 'feature_dim'):
            feature_dim = list(net.children())[-1].in_features
        else:
            feature_dim = net.feature_dim

        self.backbone.fc = MLP(feature_dim, projection_size, projection_hidden_size)  # projector

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size, predictor_network)
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.stop_gradient = stop_gradient
        self.has_predictor = has_predictor

        # debug purpose
        # self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size))
        # self.reset_moving_average()

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.backbone)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.backbone)

    def forward(self, image_one, image_two):
        p1, online_pred_one = self.backbone(image_one)
        p2, online_pred_two = self.backbone(image_two)

        if self.has_predictor:
            online_pred_one = self.online_predictor(online_pred_one)
            online_pred_two = self.online_predictor(online_pred_two)

        if self.stop_gradient:
            with torch.no_grad():
                if self.target_encoder is None:
                    self.target_encoder = self._get_target_encoder()
                _, target_proj_one = self.target_encoder(image_one)
                _, target_proj_two = self.target_encoder(image_two)

                target_proj_one = target_proj_one.detach()
                target_proj_two = target_proj_two.detach()

        else:
            if self.target_encoder is None:
                self.target_encoder = self._get_target_encoder()
            _, target_proj_one = self.target_encoder(image_one)
            _, target_proj_two = self.target_encoder(image_two)

        loss_one = byol_loss_fn(online_pred_one, target_proj_two)
        loss_two = byol_loss_fn(online_pred_two, target_proj_one)
        loss = loss_one + loss_two

        return p1, p2, loss.mean()


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096, num_layer=TwoLayer):
        super().__init__()
        self.in_features = dim
        if num_layer == OneLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, projection_size),
            )
        elif num_layer == TwoLayer:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, projection_size),
            )
        else:
            raise NotImplementedError(f"Not defined MLP: {num_layer}")

    def forward(self, x):
        return self.net(x)
