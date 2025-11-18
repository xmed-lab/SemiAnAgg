# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch


class GaussianWeightingHook():
    """
    SoftMatch learnable truncated Gaussian weighting
    """

    def __init__(self, num_classes, n_sigma=2, momentum=0.9):
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.m = momentum

        # initialize Gaussian mean and variance
        self.prob_max_mu_t = torch.ones((self.num_classes)) / self.num_classes
        self.prob_max_var_t = torch.ones((self.num_classes))

    @torch.no_grad()
    def update(self, class_centeroids_all):
        prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
        prob_max_var_t = torch.ones_like(self.prob_max_var_t)
        for i in range(self.num_classes):
            prob = class_centeroids_all[:, i]
            if len(prob) > 1:
                prob_max_mu_t[i] = torch.mean(prob)
                prob_max_var_t[i] = torch.var(prob, unbiased=True)
        self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
        self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
        return

    @torch.no_grad()
    def masking(self, class_centeroids_all):
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(class_centeroids_all.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(class_centeroids_all.device)

        self.update(class_centeroids_all)
        # compute weight
        class_centeroids_all_w = torch.zeros_like(class_centeroids_all)

        for i in range(self.num_classes):
            mu = self.prob_max_mu_t[i]
            var = self.prob_max_var_t[i]
            mask = torch.exp(
                -((torch.clamp(class_centeroids_all[:, i] - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
            class_centeroids_all_w[:, i] = mask
        return class_centeroids_all_w
