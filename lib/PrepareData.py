import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets


def calculate_img_stats(dataset):
    imgs_ = torch.stack([img for img, _ in dataset], dim=3)
    imgs_ = imgs_.view(3, -1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    return imgs_mean, imgs_std


def prepare_loader(data_dir,
                   train_transform,
                   valid_transform,
                   test_transforms,
                   batch_size=20,
                   num_workers=0,
                   valid_size=0.2):
    # data set
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)

    valid_data = datasets.ImageFolder(data_dir + '/train', transform=valid_transform)

    test_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    print("Train size:{}".format(num_train))
    print("Valid size:{}".format(len(valid_data)))
    print("Test size:{}".format(len(test_data)))

    # mix data
    # index of num of train
    indices = list(range(num_train))
    # random the index
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    # divied into two part
    train_idx, valid_idx = indices[split:], indices[:split]

    # define the sampler
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return [train_loader, valid_loader, test_loader]
