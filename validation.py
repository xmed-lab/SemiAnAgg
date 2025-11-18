# import os
# import sys
#
# import shutil
# import argparse
# import logging
# import time
# import random
# import numpy as np
# import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.metrics import compute_metrics_test
from utils.utils_SimPLE import get_class_compose


def epochVal_metrics_test(model, dataLoader, model_type, n_classes,return_centeroids=False):
    training = model.training
    model.eval()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()

    gt_study = {}
    pred_study = {}
    studies = []
    class_centeroids = torch.zeros(n_classes, 512).cuda()
    class_count = torch.tensor(np.array(get_class_compose(dataLoader, n_classes))).cuda()

    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            feature, output = model(image)
            study = study.tolist()
            output = F.softmax(output, dim=1)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])
            if return_centeroids:
                labels = label.view(label.size(0), 1).expand(-1, feature.size(1))
                semantic_proj = nn.functional.normalize(feature, dim=-1)  # already normalized
                unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
                batch_unique_target = unique_labels[:, 0].long()
                res = torch.zeros_like(class_centeroids, dtype=torch.float).scatter_add_(0,
                                                                                         labels.type(torch.int64),
                                                                                         semantic_proj.detach())
                res[batch_unique_target] = res[batch_unique_target] / labels_count.float().unsqueeze(1)
                target_class_total_count = class_count[batch_unique_target]

                class_centeroids[batch_unique_target] += (labels_count / target_class_total_count).unsqueeze(1) * \
                                                         res[batch_unique_target]

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
        # gt=F.one_hot(gt.to(torch.int64).squeeze())
        # AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred,  thresh=thresh, competition=True)
        AUROCs, Accus, Pre, Recall, b_Accus = compute_metrics_test(gt, pred, n_classes=n_classes)

    model.train(training)
    if return_centeroids:
        class_centeroids = nn.functional.normalize(class_centeroids, dim=-1)
        class_centeroids = class_centeroids.detach()
    return AUROCs, Accus, Pre, Recall, b_Accus,class_centeroids, class_count  # ,all_features.cpu(),all_labels.cpu()#, Senss, Specs, pre,F1
