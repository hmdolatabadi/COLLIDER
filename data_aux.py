import torch.utils.data as data
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from skimage.transform import resize


class CustomTensorDataset(data.Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors   = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index].long()

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class CustomIndexedTensorDataset(data.Dataset):
    """
    TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None, pois_idx=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors   = tensors
        self.transform = transform
        self.pois_idx  = pois_idx

        self.whole_tensors = (self.tensors[0].clone(), self.tensors[1].clone())

    def switch_data(self):
        self.tensors = self.whole_tensors

    def adjust_base_indx_tmp(self, idx):
        new_data     = self.whole_tensors[0][idx, ...]
        new_targets  = self.whole_tensors[1][idx, ...]
        self.tensors = (new_data, new_targets)

    def estimate_label_acc(self, idx):
        if self.pois_idx is not None:
            intersect = np.intersect1d(np.array(idx), self.pois_idx.ravel())
            label_acc = 1 - len(intersect) / len(self.pois_idx.ravel())
        else:
            label_acc = 1
        return label_acc

    def fetch(self, targets):
        whole_targets_np = np.array(self.tensors[1])
        uniq_targets     = np.unique(whole_targets_np)

        idx_dict = {}
        for uniq_target in uniq_targets:
            idx_dict[uniq_target] = np.where(whole_targets_np == uniq_target)[0]

        idx_list = []
        for target in targets:
            idx_list.append(np.random.choice(idx_dict[target.item()], 1))

        idx_list = np.array(idx_list).flatten()
        imgs     = []
        for idx in idx_list:
            img = self.tensors[0][idx]
            img = self.transform(img)
            imgs.append(img[None, ...])

        train_data = torch.cat(imgs, dim=0)

        return train_data


    def LID_fetch(self, indices):

        imgs     = []
        for idx in indices:
            img = self.tensors[0][idx]
            img = self.transform(img)
            imgs.append(img[None, ...])

        train_data = torch.cat(imgs, dim=0)

        return train_data


    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index].long()

        return x, y, index

    def __len__(self):
        return self.tensors[0].size(0)


def get_dataset(root,
                dataset,
                attack_type,
                injection_rate,
                partition,
                data_transform,
                valid_frac=0.04,
                indexed=True,
                lid_batch_size=100,
                seed=0):

    if dataset == 'cifar10':
        normalizer = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        img_size = 32
        pad = 4

    elif dataset == 'svhn':
        normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img_size = 32
        pad = 4

    elif dataset == 'gtsrb' or dataset == 'imagenet12':
        normalizer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img_size = 224
        pad = 28
    else:
        raise ValueError('No such dataset!')

    if data_transform == 'train':

        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.RandomCrop(img_size, padding=pad),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalizer,
                                        ])

    else:
        transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.ToTensor(),
                                        normalizer,
                                        ])

    if partition == 'train':
        if attack_type != 'no_backdoor':
            dataset = torch.load(os.path.join(root,
                                              f'./{dataset}_{attack_type}_train_{seed}_{lid_batch_size}_{injection_rate}.pth'))
            data, labels, perm = dataset['data'], dataset['targets'], dataset['pois_idx']

        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
            dataset      = torch.load(os.path.join(root, f'{dataset}_train.pth'))
            data, labels = dataset['data'], dataset['targets']
            num_train    = data.shape[0]
            indices      = torch.randperm(num_train).tolist()
            valid_size   = int(np.floor(valid_frac * num_train))
            train_idx    = indices[valid_size:]
            data, labels = data[train_idx], labels[train_idx]

    elif partition == 'val':
        if attack_type != 'no_backdoor':
            dataset = torch.load(os.path.join(root,
                                              f'./{dataset}_{attack_type}_val_{seed}_{lid_batch_size}_{injection_rate}.pth'))
            data, labels = dataset['data'], dataset['targets']

        else:
            np.random.seed(seed)
            torch.manual_seed(seed)
            dataset      = torch.load(os.path.join(root, f'{dataset}_train.pth'))
            data, labels = dataset['data'], dataset['targets']
            num_train    = data.shape[0]
            indices      = torch.randperm(num_train).tolist()
            valid_size   = int(np.floor(valid_frac * num_train))
            valid_idx    = indices[:valid_size]
            data, labels = data[valid_idx], labels[valid_idx]
    else:
        if  attack_type == 'sig':
            dataset = torch.load(os.path.join(root,
                                              f'./{dataset}_{attack_type}_test_{seed}_{lid_batch_size}_{injection_rate}.pth'))
            data, labels = dataset['data'], dataset['targets']

        elif attack_type == 'cl':
            dataset = torch.load(os.path.join(root,
                                              f'./cifar10_{attack_type}_test_full_intensity.pth'))
            data, labels = dataset['data'], dataset['targets']

        elif attack_type == 'refool' or attack_type == 'htba' or attack_type == 'sticker' or attack_type == 'badnets':
            dataset = torch.load(os.path.join(root,
                                              f'./{dataset}_{attack_type}_test.pth'))
            data, labels = dataset['data'], dataset['targets']

        elif attack_type == 'no_backdoor':
            dataset = torch.load(os.path.join(root, f'{dataset}_val.pth'))
            data, labels = dataset['data'], dataset['targets']

    if indexed:

        dataset = CustomIndexedTensorDataset(tensors=(data, labels),
                                             transform=transform,
                                             pois_idx=perm if partition=='train' and attack_type!='no_backdoor' else None)

    else:

        dataset = CustomTensorDataset(tensors=(data, labels),
                                      transform=transform)

    return dataset