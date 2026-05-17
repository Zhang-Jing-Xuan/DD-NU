# python tsneVis.py -d latent_debug --nclass 10 -n convnet --depth 3  --pt_from 2 --seed 2023

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from openTSNE import TSNE
from examples import utils
from data import ClassDataLoader, ClassMemDataLoader,TensorDataset
from train import define_model, train_epoch, validate
from condense_latent import load_resized_data, diffaug


def main(args, logger, repeat=1):
    data, target = torch.load(os.path.join("/root/autodl-tmp/results/dit-distillation/imagenet-100-woof-play/", 'data.pt'))
    train_transform=None
    data=torch.stack(data).squeeze(1)
    target=torch.Tensor(target)
    x=[]
    target_b=[[i]*(target.shape[0]//10) for i in range(10)]
    for i in range(len(target_b)): x=x+target_b[i]
    x=torch.Tensor(x)
    trainset = TensorDataset(data, x, train_transform)
    val_loader = TensorDataset(data, x, train_transform)
    trainset.nclass = 10
    val_loader.nclass = 10
    # trainset, val_loader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
        val_loader = ClassMemDataLoader(val_loader, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=False)
    nclass = trainset.nclass
    # _, aug_rand = diffaug(args)
    dd_data, dd_target = torch.load(os.path.join("/root/autodl-tmp/results/dit-distillation/imagenet-100-woof-play/", 'data.pt'))
    for i in range(repeat):
        logger(f"\nRepeat: {i + 1}/{repeat}")
        model = define_model(args, nclass, logger)
        checkpoint = torch.load(
            "/root/autodl-tmp/MinimaxDiffusion/pretrained_models/latent/conv3in_cut_seed_0_lr_0.01_aug_color_crop_cutout/checkpoint_best_0.pth.tar"
        )
        model.load_state_dict(checkpoint)
        model.eval()
        # pretrain(model, loader_real, val_loader, logger, i, args)
        pretrain(model, loader_real, dd_data, dd_target, args)


def pretrain(model, loader_real, dd_data, dd_target, args):
    model = model.cuda()
    save_target, save_features = None, None
    # for i, (input, target) in enumerate(loader_real):
    #     input = input.cuda()
    #     target = target.cuda()
    #     output, features = model(input, return_features=True)
    #     if save_target == None:
    #         save_target, save_features = target.detach().cpu(
    #         ), features.detach().cpu()
    #     else:
    #         save_target = torch.cat([save_target, target.detach().cpu()])
    #         save_features = torch.cat([save_features, features.detach().cpu()])
    # print(save_target.shape, save_features.shape)
    dd_data, dd_target=torch.load(os.path.join("/root/autodl-tmp/results/dit-distillation/imagenet-1000-woof-play/", 'data.pt'))
    # dd_data, dd_target = decode(dd_data,dd_target)
    dd_data, dd_target = torch.stack(dd_data).squeeze(1),torch.Tensor(dd_target)
    dd_data, dd_target = dd_data.cuda(), dd_target.cuda()
    dd_output, dd_features = model(dd_data, return_features=True)
    save_dd_target, save_dd_features = dd_target.detach().cpu(
            ), dd_features.detach().cpu()
    ###
    save_target = save_dd_target
    save_features = save_dd_features

    tsne = TSNE(perplexity=50,
                metric="euclidean",
                n_jobs=8,
                random_state=42,
                verbose=True)
    embedding_train = tsne.fit(save_features)
    utils.plot(embedding_train, save_target, name=10,colors=utils.MACOSKO_COLORS)
    utils.plot(embedding_train, save_target, name=20,colors=utils.MACOSKO_COLORS)
    utils.plot(embedding_train, save_target, name=50,colors=utils.MACOSKO_COLORS)
    utils.plot(embedding_train, save_target, name=100,colors=utils.MACOSKO_COLORS)


if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn

    assert args.pt_from > 0, "set args.pt_from positive! (epochs for pretraining)"

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    args.save_dir = f"./pretrained_models/{args.datatag}/{args.modeltag}{args.tag}_seed_{args.seed}_lr_{args.lr}_aug_{args.aug_type}"
    os.makedirs(args.save_dir, exist_ok=True)

    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    logger(f"Seed: {args.seed}")
    logger(f"Lr: {args.lr}")
    logger(f"Aug-type: {args.aug_type}")
    main(args, logger, args.repeat)