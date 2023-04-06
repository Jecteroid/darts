# coding: utf-8
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
from tools import preproc
from torchvision import datasets
import torchvision
from pathlib import Path


def get_data(dataset, data_path, cutout_length, validation):
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
        trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
        trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
        trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)
    elif dataset == 'custom':
        # Setup path to data folder
        train_path = data_path + '/train'

        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),

            torchvision.transforms.ToTensor()
        ])

        trn_data = datasets.ImageFolder(root=train_path,
                                        transform=data_transform,
                                        target_transform=None)

        n_classes = len(trn_data.classes)
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    shape = trn_data.data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]

    if validation:
        if dataset != 'custom':
            ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))
        else:
            train_path = data_path + '/test'

            data_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(p=0.5),

                torchvision.transforms.ToTensor()
            ])

            test_data = datasets.ImageFolder(root=train_path,
                                             transform=data_transform,
                                             target_transform=None)

            ret.append(test_data)

    return ret


def get_logger(file_path):
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in Mb"""
    n_params = sum(np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    if target.ndimension() > 1:
        terget = terget.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
