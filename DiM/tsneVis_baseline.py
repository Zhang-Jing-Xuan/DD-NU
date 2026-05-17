# python tsneVis.py --seed 3407
import torch
import torch.nn as nn
import torch.optim as optim
from openTSNE import TSNE
import os
import numpy as np
from examples import utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN

import torch.nn.functional as F
from math import ceil

from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from misc.augment import DiffAug
import misc


def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.data == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.data == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.data == 'svhn':
        train_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                      split='train',
                                      transform=transforms.ToTensor())
        train_dataset.targets = train_dataset.labels

        normalize = transforms.Normalize(mean=MEANS['svhn'], std=STDS['svhn'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                    split='test',
                                    transform=transform_test)
        train_dataset.nclass = 10

    elif args.data == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.data == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    

    val_loader = MultiEpochsDataLoader(val_dataset,
                                       batch_size=args.batch_size // 2,
                                       shuffle=False,
                                       persistent_workers=True,
                                       num_workers=4)

    assert train_dataset[0][0].shape[-1] == val_dataset[0][0].shape[-1]  # width check

    return train_dataset, val_loader

def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)

def diffaug(args, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type
    normalize = misc.utils.Normalize(mean=MEANS[args.data], std=STDS[args.data], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand

def str2bool(v):
    """Cast string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def define_model(args, num_classes, e_model=None):
    '''Obtain model for training and validating
    '''
    if e_model:
        model = e_model
    else:
        model = args.match_model

    if args.data == 'mnist' or args.data == 'fashion':
        nch = 1
    else:
        nch = 3

    if model == 'convnet':
        return CN.ConvNet(num_classes, channel=nch)
    elif model == 'resnet10':
        return RN.ResNet(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18':
        return RN.ResNet(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34':
        return RN.ResNet(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50':
        return RN.ResNet(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101':
        return RN.ResNet(args.data, 101, num_classes, nch=nch)
    elif model == 'resnet10_ap':
        return RNAP.ResNetAP(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18_ap':
        return RNAP.ResNetAP(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34_ap':
        return RNAP.ResNetAP(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50_ap':
        return RNAP.ResNetAP(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101_ap':
        return RNAP.ResNetAP(args.data, 101, num_classes, nch=nch)
    elif model == 'densenet':
        return DN.densenet_cifar(num_classes)


def main(args, logger, repeat=1):
    trainset, val_loader = load_resized_data(args)
    loader_real = ClassMemDataLoader(args,trainset, batch_size=64)
    # if args.load_memory:
    #     loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    # else:
    #     loader_real = ClassDataLoader(trainset,
    #                                   batch_size=args.batch_real,
    #                                   num_workers=args.workers,
    #                                   shuffle=True,
    #                                   pin_memory=True,
    #                                   drop_last=False)
    nclass = trainset.nclass
    _, aug_rand = diffaug(args)
    dd_path = "/home/dd/DL/DataDistillation/DiM/logs/test_baseline/data_best.pt"
    dd_data, dd_target = torch.load(dd_path)
    for i in range(repeat):
        logger(f"\nRepeat: {i + 1}/{repeat}")
        model = define_model(args, nclass, "convnet")
        checkpoint = torch.load(
            "/home/dd/DL/DataDistillation/Acc-DD/pretrained_models/cifar10/conv3in_cut_seed_2023_lr_0.01_aug_color_crop_cutout/checkpoint_best_0.pth.tar"
        )
        model.load_state_dict(checkpoint)
        model.eval()
        pretrain(model, loader_real, dd_data, dd_target, aug_rand, args)


def pretrain(model, loader_real, dd_data, dd_target, aug_rand, args):
    model = model.cuda()
    save_target, save_features = None, None
    for i, (input, target) in enumerate(loader_real):
        input = input.cuda()
        target = target.cuda()
        output, features = model(input, return_features=True)
        if save_target == None:
            save_target, save_features = target.detach().cpu(
            ), features.detach().cpu()
        else:
            save_target = torch.cat([save_target, target.detach().cpu()])
            save_features = torch.cat([save_features, features.detach().cpu()])
    print(save_target.shape, save_features.shape)
    dd_data, dd_target = dd_data.cuda(), dd_target.cuda()
    dd_output, dd_features = model(dd_data, return_features=True)
    save_dd_target, save_dd_features = dd_target.detach().cpu(
            ), dd_features.detach().cpu()
    save_target = torch.cat([save_target, save_dd_target])
    save_features = torch.cat([save_features, save_dd_features])
    tsne = TSNE(perplexity=50,
                metric="euclidean",
                n_jobs=8,
                random_state=42,
                verbose=True)
    embedding_train = tsne.fit(save_features)
    utils.plot(embedding_train, save_target, colors=utils.MACOSKO_COLORS)
    # embedding_train = tsne.fit(save_dd_features)
    # utils.plot(embedding_train, save_dd_target, colors=utils.MACOSKO_COLORS)


if __name__ == '__main__':
    import shutil
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipc', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--epochs-eval', type=int, default=1000)
    parser.add_argument('--epochs-match', type=int, default=100)
    parser.add_argument('--epochs-match-train', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eval-lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--match-coeff', type=float, default=0.001)
    parser.add_argument('--match-model', type=str, default='convnet')
    parser.add_argument('--eval-model', type=str, nargs='+', default=['convnet'])
    parser.add_argument('--dim-noise', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--test-interval', type=int, default=200)
    parser.add_argument('--fix-disc', action='store_true', default=False)

    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--logs-dir', type=str, default='./logs/')
    parser.add_argument('--pretrain-weight', type=str, default='./logs/')
    parser.add_argument('--match-aug', action='store_true', default=False)
    parser.add_argument('--aug-type', type=str, default='color_crop_cutout')
    parser.add_argument('--mixup-net', type=str, default='cut')
    parser.add_argument('--metric', type=str, default='l1')
    parser.add_argument('--bias', type=str2bool, default=False)
    parser.add_argument('--fc', type=str2bool, default=False)
    parser.add_argument('--mix-p', type=float, default=-1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--tag', type=str, default='test')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('-a',
                    '--aug_type',
                    type=str,
                    default='color_crop_cutout',
                    help='augmentation strategy for condensation matching objective')
    args = parser.parse_args()

    from misc.utils import Logger
    import torch.backends.cudnn as cudnn

    # assert args.pt_from > 0, "set args.pt_from positive! (epochs for pretraining)"

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    args.save_dir = f"./Visualize/{args.tag}_seed_{args.seed}"
    os.makedirs(args.save_dir, exist_ok=True)

    # cur_file = os.path.join(os.getcwd(), __file__)
    # shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    logger(f"Seed: {args.seed}")
    logger(f"Aug-type: {args.aug_type}")
    main(args, logger, 1)