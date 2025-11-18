# encoding: utf-8
"""
Read images and corresponding labels.
"""
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

N_CLASSES = 10


class CheXpertDataset(Dataset):
    def __init__(self, dataset_type, data_np, label_np, pre_w, pre_h, lab_trans=None, un_trans_wk=None, data_idxs=None,
                 is_labeled=False,
                 is_testing=False):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CheXpertDataset, self).__init__()

        self.images = data_np
        self.labels = label_np
        self.is_labeled = is_labeled
        self.dataset_type = dataset_type
        self.is_testing = is_testing

        self.resize = transforms.Compose([transforms.Resize((pre_w, pre_h))])
        if not is_testing:
            if is_labeled == True:
                self.transform = lab_trans
            else:
                self.data_idxs = data_idxs
                self.weak_trans = un_trans_wk
        else:
            self.transform = lab_trans

        print('Total # images:{}, labels:{}'.format(len(self.images), len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        if self.dataset_type == 'skin':
            img_path = self.images[index]
            image = Image.open(img_path).convert('RGB')
        else:
            image = Image.fromarray(self.images[index]).convert('RGB')

        image_resized = self.resize(image)
        label = self.labels[index]

        if not self.is_testing:
            if self.is_labeled == True:
                if self.transform is not None:
                    image = self.transform(image_resized).squeeze()
                    # image=image[:,:224,:224]
                    return index, image, torch.FloatTensor([label])
            else:
                if self.weak_trans and self.data_idxs is not None:
                    weak_aug = self.weak_trans(image_resized)
                    idx_in_all = self.data_idxs[index]

                    for idx in range(len(weak_aug)):
                        weak_aug[idx] = weak_aug[idx].squeeze()
                    return index, weak_aug, torch.FloatTensor([label])
        else:
            image = self.transform(image_resized)
            return index, image, torch.FloatTensor([label])
            # return index, weak_aug, strong_aug, torch.FloatTensor([label])

    def __len__(self):
        return len(self.labels)


import torchvision.transforms.functional as TF


def image_rot(image, angle):
    image = TF.rotate(image, angle)
    return image


class TransformTwice:
    def __init__(self, transform, transform_prime, is_orchestra=False, return_two=False):
        self.transform = transform
        self.transform_prime = transform_prime
        self.is_orchestra = is_orchestra
        self.return_two=return_two
    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform_prime(inp)

        if self.is_orchestra:
            n = random.random()
            angle = 0 if n <= 0.25 else 1 if n <= 0.5 else 2 if n <= 0.75 else 3
            x1 = self.transform(inp)
            x2 = self.transform_prime(inp)
            x3 = image_rot(self.transform(inp), 90 * angle)
            return [x1, x2, x3, torch.tensor([angle])]
        if self.return_two:
            out_3 = self.transform_prime(inp)
            return [out1, out2, out_3]

        return [out1, out2]
