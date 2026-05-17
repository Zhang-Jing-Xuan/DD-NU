# python condense_latent_Only_1k.py --reproduce -d latent -f 1 --ipc 10 -n convnet --depth 3 --model_path "/root/autodl-tmp/MinimaxDiffusion/pretrained_models/latent_1k/conv3in_cut_seed_2023_lr_0.1_aug_color_crop_cutout/" --niter 150 --inner_loop 10 --tag grad10_1k_Only_DM --val_interval 10
# python condense_latent_original_1k.py --reproduce -d latent -f 1 --ipc 10 -n convnet --depth 3 --model_path "/root/autodl-tmp/MinimaxDiffusion/pretrained_models/latent_1k/conv3in_cut_seed_2023_lr_0.1_aug_color_crop_cutout/" --niter 100 --inner_loop 10 --tag grad10_750_1k_25% --val_interval 5
# python condense_latent.py --reproduce -d latent -f 1 --ipc 10 -n convnet --depth 3 --model_path "/root/autodl-tmp/MinimaxDiffusion/pretrained_models/latent_woof/conv3in_cut_seed_2023_lr_0.01_aug_color_crop_cutout/" --niter 1000 --inner_loop 10 --tag grad10_1000_4
# python condense_latent.py --reproduce -d latent -f 1 --ipc 50 -n convnet --depth 3 --model_path "/root/autodl-tmp/MinimaxDiffusion/pretrained_models/latent_woof/conv3in_cut_seed_2023_lr_0.01_aug_color_crop_cutout/" --niter 1000 --inner_loop 10 --tag grad50_1000_4 
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from data import transform_imagenet, transform_cifar, transform_svhn, transform_mnist, transform_fashion
from data import TensorDataset, ImageFolder, save_img
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import MEANS, STDS
from train import define_model, train_epoch
# from test import test_data, load_ckpt
from misc.augment import DiffAug
from misc import utils
from math import ceil
import glob
import random
from weight_perturbation import setup_directions, get_weights, set_weights, set_states, setup_directions_random
import copy
import math
from data import load_data
from misc.utils import AverageMeter, accuracy, get_time
import time
import copy

class Synthesizer():
    """Condensed data class
    """

    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel
        self.size = (hs, ws)
        self.device = device

        self.data = torch.randn(size=(self.nclass * self.ipc, self.nchannel, hs, ws),
                                dtype=torch.float,
                                requires_grad=True,
                                device=self.device)
        # self.data.data = torch.clamp(self.data.data / 4 + 0.5, min=0., max=1.)
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.long,
                                    requires_grad=False,
                                    device=self.device).view(-1)
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(self.data.shape[0]):
            self.cls_idx[self.targets[i]].append(i)

        print("\nDefine synthetic data: ", self.data.shape)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type
        self.resize = nn.Upsample(size=self.size, mode='bilinear')
        print(f"Factor: {self.factor} ({self.decode_type})")

    def init(self, loader, init_type='noise'):
        """Condensed data initialization
        """
        if init_type == 'random':
            print("Random initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc)
                # noise = torch.randn_like(img.data) 
                # noisy_image = img + noise
                # self.data.data[self.ipc * c:self.ipc * (c + 1)] = noisy_image.to(self.device)
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img.data.to(self.device)
                
        elif init_type == 'exist':
            img, _ = torch.load("/root/autodl-tmp/MinimaxDiffusion/results/latent/conv3in_grad10_1k_Only_grad_mse_pt2_nd2000_inloop10_cut_nlr0.1_wd0.0001_niter300_factor1_lr0.05_b_real128_random_ipc10/data_best.pt")
            for c in range(self.nclass):
                self.data.data[self.ipc * c:self.ipc * (c + 1)] = img[self.ipc * c:self.ipc * (c + 1)].data.to(self.device)

        elif init_type == 'mix':
            # print("Mixed initialize synset")
            for c in range(self.nclass):
                img, _ = loader.class_sample(c, self.ipc * self.factor ** 2)
                img = img.data.to(self.device)

                s = self.size[0] // self.factor
                remained = self.size[0] % self.factor
                k = 0
                n = self.ipc

                h_loc = 0
                for i in range(self.factor):
                    h_r = s + 1 if i < remained else s
                    w_loc = 0
                    for j in range(self.factor):
                        w_r = s + 1 if j < remained else s
                        img_part = F.interpolate(img[k * n:(k + 1) * n], size=(h_r, w_r))
                        self.data.data[n * c:n * (c + 1), :, h_loc:h_loc + h_r,
                        w_loc:w_loc + w_r] = img_part
                        w_loc += w_r
                        k += 1
                    h_loc += h_r

        elif init_type == 'noise':
            pass

    def parameters(self):
        parameter_list = [self.data]
        return parameter_list

    def subsample(self, data, target, max_size=-1):
        if (data.shape[0] > max_size) and (max_size > 0):
            indices = np.random.permutation(data.shape[0])
            data = data[indices[:max_size]]
            target = target[indices[:max_size]]

        return data, target

    def decode_zoom(self, img, target, factor):
        """Uniform multi-formation
        """
        h = img.shape[-1]
        remained = h % factor
        if remained > 0:
            img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
        s_crop = ceil(h / factor)
        n_crop = factor ** 2

        cropped = []
        for i in range(factor):
            for j in range(factor):
                h_loc = i * s_crop
                w_loc = j * s_crop
                cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
        cropped = torch.cat(cropped)
        data_dec = self.resize(cropped)
        target_dec = torch.cat([target for _ in range(n_crop)])

        return data_dec, target_dec

    def decode_zoom_multi(self, img, target, factor_max):
        """Multi-scale multi-formation
        """
        data_multi = []
        target_multi = []
        for factor in range(1, factor_max + 1):
            decoded = self.decode_zoom(img, target, factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

        return torch.cat(data_multi), torch.cat(target_multi)

    def decode_zoom_bound(self, img, target, factor_max, bound=128):
        """Uniform multi-formation with bounded number of synthetic data
        """
        bound_cur = bound - len(img)
        budget = len(img)

        data_multi = []
        target_multi = []

        idx = 0
        decoded_total = 0
        for factor in range(factor_max, 0, -1):
            decode_size = factor ** 2
            if factor > 1:
                n = min(bound_cur // decode_size, budget)
            else:
                n = budget

            decoded = self.decode_zoom(img[idx:idx + n], target[idx:idx + n], factor)
            data_multi.append(decoded[0])
            target_multi.append(decoded[1])

            idx += n
            budget -= n
            decoded_total += n * decode_size
            bound_cur = bound - decoded_total - budget

            if budget == 0:
                break

        data_multi = torch.cat(data_multi)
        target_multi = torch.cat(target_multi)
        return data_multi, target_multi

    def decode(self, data, target, bound=128):
        """Multi-formation
        """
        if self.factor > 1:
            if self.decode_type == 'multi':
                data, target = self.decode_zoom_multi(data, target, self.factor)
            elif self.decode_type == 'bound':
                data, target = self.decode_zoom_bound(data, target, self.factor, bound=bound)
            else:
                data, target = self.decode_zoom(data, target, self.factor)

        return data, target

    def sample(self, c, max_size=128):
        """Sample synthetic data per class
        """
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)
        data = self.data[idx_from:idx_to]
        target = self.targets[idx_from:idx_to]

        data, target = self.decode(data, target, bound=max_size)
        data, target = self.subsample(data, target, max_size=max_size)
        return data, target

    def loader(self, args, augment=True):
        """Data loader for condensed data
        """
        if args.dataset == 'imagenet':
            train_transform, _ = transform_imagenet(augment=augment,
                                                    from_tensor=True,
                                                    size=0,
                                                    rrc=args.rrc,
                                                    rrc_size=self.size[0])
        elif args.dataset[:5] == 'cifar':
            train_transform, _ = transform_cifar(augment=augment, from_tensor=True)
        elif args.dataset == 'svhn':
            train_transform, _ = transform_svhn(augment=augment, from_tensor=True)
        elif args.dataset == 'mnist':
            train_transform, _ = transform_mnist(augment=augment, from_tensor=True)
        elif args.dataset == 'fashion':
            train_transform, _ = transform_fashion(augment=augment, from_tensor=True)

        data_dec = []
        target_dec = []
        for c in range(self.nclass):
            idx_from = self.ipc * c
            idx_to = self.ipc * (c + 1)
            data = self.data[idx_from:idx_to].detach()
            target = self.targets[idx_from:idx_to].detach()
            data, target = self.decode(data, target)

            data_dec.append(data)
            target_dec.append(target)

        data_dec = torch.cat(data_dec)
        target_dec = torch.cat(target_dec)

        train_dataset = TensorDataset(data_dec.cpu(), target_dec.cpu(), train_transform)

        print("Decode condensed data: ", data_dec.shape)
        nw = 0 if not augment else args.workers
        train_loader = MultiEpochsDataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             persistent_workers=nw > 0)
        return train_loader

    # def test(self, args, val_loader, logger, bench=True):
    #     """Condensed data evaluation
    #     """
    #     vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mae").to("gpu")
    #     loader = self.loader(args, args.augment)
    #     best_acc, last_acc = test_data(args, loader, val_loader, repeat=3, test_resnet=False, logger=logger)
    #     return best_acc, last_acc

        # if bench and not (args.dataset in ['mnist', 'fashion']):
        #     test_data(args, loader, val_loader, test_resnet=True, logger=logger)


def load_resized_data(args):
    """Load original training data (fixed spatial size and without augmentation) for condensation
    """
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, train=True, transform=transforms.ToTensor())
        normalize = transforms.Normalize(mean=MEANS['cifar10'], std=STDS['cifar10'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR10(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir,
                                          train=True,
                                          transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['cifar100'], std=STDS['cifar100'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
        val_dataset = datasets.CIFAR100(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 100

    elif args.dataset == 'svhn':
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

    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(args.data_dir, train=True, transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['mnist'], std=STDS['mnist'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'fashion':
        train_dataset = datasets.FashionMNIST(args.data_dir,
                                              train=True,
                                              transform=transforms.ToTensor())

        normalize = transforms.Normalize(mean=MEANS['fashion'], std=STDS['fashion'])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        val_dataset = datasets.FashionMNIST(args.data_dir, train=False, transform=transform_test)
        train_dataset.nclass = 10

    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.imagenet_dir, 'train')
        valdir = os.path.join(args.imagenet_dir, 'val')

        # We preprocess images to the fixed size (default: 224)
        resize = transforms.Compose([
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.PILToTensor()
        ])

        if args.load_memory:  # uint8
            transform = None
            load_transform = resize
        else:
            transform = transforms.Compose([resize, transforms.ConvertImageDtype(torch.float)])
            load_transform = None

        _, test_transform = transform_imagenet(size=args.size)
        train_dataset = ImageFolder(traindir,
                                    transform=transform,
                                    nclass=args.nclass,
                                    phase=args.phase,
                                    seed=args.dseed,
                                    load_memory=args.load_memory,
                                    load_transform=load_transform)
        val_dataset = ImageFolder(valdir,
                                  test_transform,
                                  nclass=args.nclass,
                                  phase=args.phase,
                                  seed=args.dseed,
                                  load_memory=False)

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
    normalize = utils.Normalize(mean=MEANS[args.dataset], std=STDS[args.dataset], device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss


def matchloss(args, img_real, img_syn, lab_real, lab_syn, model,it):
    """Matching losses (feature or gradient)
    """
    loss = None

    if args.match == 'feat':
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, args.idx_from, args.idx_to)
        feat = model.get_feature(img_syn, args.idx_from, args.idx_to)

        for i in range(len(feat)):
            loss = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))

    elif args.match == 'grad':
#         criterion = nn.CrossEntropyLoss()

#         output_real = model(img_real)
#         lab_real=lab_real.type(torch.long) ###
#         loss_real = criterion(output_real, lab_real)
#         g_real = torch.autograd.grad(loss_real, model.parameters())
#         g_real = list((g.detach() for g in g_real))

#         output_syn = model(img_syn)
#         loss_syn = criterion(output_syn, lab_syn)
#         g_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

#         for i in range(len(g_real)):
#             if (len(g_real[i].shape) == 1) and not args.bias:  # bias, normliazation
#                 continue
#             if (len(g_real[i].shape) == 2) and not args.fc:
#                 continue

#             loss = add_loss(loss, dist(g_real[i], g_syn[i], method=args.metric))
        
        with torch.no_grad():
            feat_tg = model.get_feature(img_real, 0, 3)
        feat = model.get_feature(img_syn, 0, 3)

        for i in range(len(feat)):
            loss_fea = add_loss(loss, dist(feat_tg[i].mean(0), feat[i].mean(0), method=args.metric))
        loss = loss_fea
        # if it>50:
        #     loss=loss+0.2*loss_fea
    return loss


def remove_prefix_checkpoint(dictionary, prefix):
    keys = sorted(dictionary.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) + 1:]
            dictionary[newkey] = dictionary.pop(key)
    return dictionary

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for param in self.model.parameters():
            if param.requires_grad:
                self.shadow["img"] = param.data.clone()

    def update(self):
        for param in self.model.parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow["img"]
                self.shadow["img"] = new_average.clone()

    def apply_shadow(self):
        for param in self.model.parameters():
            if param.requires_grad:
                self.backup["img"] = param.data
                param.data = self.shadow["img"]

    def restore(self):
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = self.backup["img"]
        self.backup = {}

def load_state(file_dir, verbose=True):
    checkpoint = torch.load(file_dir)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = remove_prefix_checkpoint(checkpoint, 'module')
    return checkpoint

def validate_imagenet(args, val_loader, model, criterion, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if logger is not None and args.verbose == True:
        logger(
            '(Test ) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg

def train_imagenet(args, model, train_loader, val_loader, logger=None):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0
    # if args.pretrained:
    #     pretrained = "{}/{}".format(args.save_dir, 'checkpoint.pth.tar')
    #     cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)
    #     # TODO: optimizer scheduler steps

    model = model.cuda()
    logger(f"Start training with base augmentation and {args.mixup} mixup")

    # Start training and validation
    for epoch in range(cur_epoch + 1, args.epochs + 1):
        acc1_tr, _, loss_tr = train_epoch(args,
                                          train_loader,
                                          model,
                                          criterion,
                                          optimizer,
                                          epoch,
                                          logger,
                                          mixup=args.mixup)
        is_best = False
        if epoch % args.epoch_print_freq == 0:
            acc1, acc5, loss_val = validate_imagenet(args, val_loader, model, criterion, epoch, logger)

            # if plotter != None:
            #     plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                if logger != None and args.verbose == True:
                    logger(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')

        if args.save_ckpt and (is_best or (epoch == args.epochs)):
            state = {
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }
            # save_checkpoint(args.save_dir, state, is_best)
        scheduler.step()

    return best_acc1, acc1

def test_imagenet(args,path):
    os.system("python train.py -d imagenet --imagenet_dir ../results/dit-distillation/imagewoof-10-minimax /root/autodl-tmp/MinimaxDiffusion/data/imagenet/     -n resnet_ap --nclass 10 --norm_type instance --ipc 10 --tag test --slct_type random --spec woof")
    args_backup=copy.deepcopy(args)
    # args.dataset="imagenet"
    # args.datatag="imagenet10"
    # args.depth=10
    # args.imagenet_dir=[path,"/root/autodl-tmp/MinimaxDiffusion/data/imagenet/"]
    # args.net_type="resnet_ap"
    # args.norm_type="instance"
    # args.tag="test"
    # args.slct_type="random"
    # args.spec="woof"
    # args.nch=3
    # args.modeltag="resnet10apin"
    # args.size=224
    # args.epochs=2000
    # _, train_loader, val_loader, nclass = load_data(args)
    # best_acc_l = []
    # acc_l = []
    # repeat=3
    # for i in range(repeat):
    #     # logger(f"Repeat: {i+1}/{repeat}")
    #     model = define_model(args, args.nclass, logger)
    #     best_acc, acc = train_imagenet(args, model, train_loader, val_loader, logger)
    #     best_acc_l.append(best_acc)
    #     acc_l.append(acc)
    #     logger(f"Repeat: {i+1}/{repeat}: {best_acc}")
    # best_acc_l=np.array(best_acc_l)
    # logger(f"Mean: {best_acc_l.mean()}, Standard: {best_acc_l.std()}")
    return args_backup

def decode_zoom(img, target, factor, size=-1):
    if size == -1:
        size = img.shape[-1]
    resize = nn.Upsample(size=size, mode='bilinear')

    h = img.shape[-1]
    remained = h % factor
    if remained > 0:
        img = F.pad(img, pad=(0, factor - remained, 0, factor - remained), value=0.5)
    s_crop = ceil(h / factor)
    n_crop = factor**2

    cropped = []
    for i in range(factor):
        for j in range(factor):
            h_loc = i * s_crop
            w_loc = j * s_crop
            cropped.append(img[:, :, h_loc:h_loc + s_crop, w_loc:w_loc + s_crop])
    cropped = torch.cat(cropped)
    data_dec = resize(cropped)
    target_dec = torch.cat([target for _ in range(n_crop)])

    return data_dec, target_dec


def decode_zoom_multi(img, target, factor_max):
    data_multi = []
    target_multi = []
    for factor in range(1, factor_max + 1):
        decoded = decode_zoom(img, target, factor)
        data_multi.append(decoded[0])
        target_multi.append(decoded[1])

    return torch.cat(data_multi), torch.cat(target_multi)


def decode_fn(data, target, factor, decode_type, bound=128):
    if factor > 1:
        if decode_type == 'multi':
            data, target = decode_zoom_multi(data, target, factor)
        else:
            data, target = decode_zoom(data, target, factor)

    return data, target


def decode(args, data, target):
    data_dec = []
    target_dec = []
    ipc = len(data) // args.nclass
    for c in range(args.nclass):
        idx_from = ipc * c
        idx_to = ipc * (c + 1)
        data_ = data[idx_from:idx_to].detach()
        target_ = target[idx_from:idx_to].detach()
        data_, target_ = decode_fn(data_,
                                   target_,
                                   args.factor,
                                   args.decode_type,
                                   bound=args.batch_syn_max)
        data_dec.append(data_)
        target_dec.append(target_)

    data_dec = torch.cat(data_dec)
    target_dec = torch.cat(target_dec)

    print("Dataset is decoded! ", data_dec.shape)
    # save_img('./results/test_dec.png', data_dec, unnormalize=False, dataname=args.dataset)
    return data_dec, target_dec

def condense(args, logger, device='cuda'):
    # trainset, val_loader = load_resized_data(args)
    # data, target = torch.load(os.path.join("/root/autodl-tmp/results/dit-distillation/imagenet-10-1000-initial-1000/", 'data.pt'))
    # data, target = torch.load(os.path.join("/root/autodl-tmp/results/dit-distillation/imagenet-10-750-initial-750", 'data.pt'))
    # data, target = torch.load("/root/autodl-tmp/results/dit-distillation/imagewoof-750.pt")
    data, target = torch.load("/root/autodl-tmp/results/dit-distillation/imagenet-1k-750/1k_dit_750.pt")
    # data_ori, target_ori = torch.load("/root/autodl-tmp/imagenet1000_full.pt")
    # data = data[750*1000*0:750*1000*1]
    # target = target[750*1000*0:750*1000*1]
    train_transform=None
    data=torch.stack(data).squeeze(1)
    target=torch.Tensor(target)
    # data_ori=torch.stack(data_ori).squeeze(1)
    # data = data_ori
    x=[]
    target_b=[[i]*(target.shape[0]//1000) for i in range(1000)] # 10
    num_imagenet1000_full=[1299, 1300, 1287, 1215, 1278, 1299, 1294, 1291, 1296, 1296, 1299, 1300, 1300, 1298, 1300, 1300, 1300, 1300, 1298, 1297, 1300, 1295, 1298, 1292, 1296, 1298, 1299, 1300, 1299, 1298, 1300, 1300, 1297, 1295, 1293, 1299, 1293, 1297, 1299, 1292, 1300, 1298, 1300, 1112, 1296, 1296, 1300, 1299, 1297, 1296, 1292, 1246, 1299, 1298, 1298, 1300, 1297, 1299, 1296, 1299, 1295, 1298, 1070, 1295, 1300, 1298, 1298, 1295, 1299, 1284, 1294, 1296, 1298, 1296, 1296, 1295, 1294, 1294, 1291, 1296, 1297, 1298, 1298, 1299, 1300, 1298, 1299, 1295, 1300, 1299, 1300, 1299, 1300, 1299, 1300, 1300, 1300, 1300, 1140, 1293, 1296, 1283, 1295, 1262, 1295, 1298, 1294, 1298, 1296, 1293, 1300, 1111, 1279, 1290, 1296, 1300, 1296, 1279, 1300, 1297, 1299, 1299, 1298, 1299, 1295, 1298, 1288, 1296, 1294, 1297, 1300, 1300, 1291, 1299, 1295, 1298, 1300, 1297, 1300, 1300, 1300, 1299, 1300, 1299, 1296, 1300, 1295, 1142, 1285, 1295, 1292, 1279, 759, 1283, 
1283, 1274, 1294, 1288, 856, 1294, 1272, 1285, 1290, 1279, 1117, 716, 1021, 753, 1287, 1288, 1287, 1271, 1283, 1295, 1271, 735, 1287, 1277, 1213, 1275, 1282, 1240, 1289, 1243, 1294, 1283, 1284, 1292, 964, 1294, 913, 1294, 1283, 1297, 1142, 1244, 1266, 1260, 1273, 
1264, 1282, 1295, 1292, 1267, 1289, 1284, 1204, 1289, 1278, 1293, 1290, 1296, 1277, 1293, 1293, 1295, 1294, 1287, 1299, 1265, 1292, 
959, 1285, 1277, 1254, 1297, 1289, 1287, 1285, 1283, 1297, 1283, 1283, 1274, 1293, 1281, 1293, 1283, 1298, 1296, 1295, 1297, 1277, 1281, 1297, 1270, 1269, 1283, 1267, 1279, 1276, 1243, 935, 1292, 1261, 1296, 1283, 1285, 1272, 1287, 1292, 1288, 1066, 1296, 1297, 1283, 1289, 1282, 750, 1276, 1289, 1297, 1293, 1295, 1299, 1298, 1295, 1299, 1298, 1284, 1292, 1277, 1289, 1283, 1289, 1275, 1289, 1293, 1295, 1276, 1295, 1291, 1291, 1295, 1299, 1294, 1289, 1289, 1296, 1287, 1300, 1300, 1289, 1299, 1300, 1295, 1295, 1298, 1295, 1297, 1291, 1299, 1293, 1296, 1285, 1298, 1298, 1300, 1296, 1298, 1300, 1300, 1300, 1300, 1298, 1300, 1300, 1288, 1294, 1300, 1289, 1293, 1294, 1295, 1286, 1202, 1298, 1290, 1292, 1300, 1235, 1284, 1294, 1288, 1294, 1272, 1289, 1287, 1282, 1295, 1291, 1300, 1299, 1292, 1291, 1279, 1288, 1287, 1295, 1281, 1291, 1280, 1279, 1291, 1297, 1279, 1257, 1256, 1282, 1294, 1294, 1297, 1293, 1294, 1292, 1297, 1297, 1296, 1296, 1296, 1297, 1298, 1297, 1296, 1297, 1290, 1287, 1296, 1290, 1293, 1157, 1298, 969, 1300, 1295, 1298, 1295, 1297, 1270, 1271, 1269, 1165, 1237, 1232, 1292, 1139, 1296, 1293, 1280, 1214, 1300, 1300, 1262, 1246, 1296, 1293, 1280, 1297, 1223, 1264, 1239, 1224, 1270, 1228, 1169, 1246, 1153, 1246, 1254, 1270, 1292, 1291, 1229, 1289, 1268, 1263, 1284, 1269, 1279, 1195, 1283, 1288, 1286, 1298, 1274, 1254, 1291, 1239, 1272, 1278, 1289, 1291, 1286, 1294, 1279, 1277, 1286, 1209, 1287, 1284, 1242, 1288, 1277, 1268, 1266, 1156, 1285, 1291, 1286, 1269, 1296, 1254, 1285, 1260, 1298, 1238, 1292, 1292, 1289, 1265, 1290, 1139, 1285, 1266, 1290, 1282, 1226, 1271, 1225, 1228, 1260, 1179, 1292, 1299, 1271, 1298, 1298, 1261, 1274, 1045, 1281, 1007, 1288, 1125, 1279, 1208, 1207, 1150, 1233, 1298, 1285, 1294, 1221, 1191, 1262, 1252, 1280, 1258, 1291, 1281, 1295, 1184, 1298, 1220, 1289, 1245, 1289, 1283, 1210, 1293, 1281, 871, 1299, 1300, 1286, 1278, 1179, 1290, 1274, 1282, 1283, 1257, 1229, 1250, 1296, 1239, 1269, 1262, 1300, 1293, 1117, 1152, 1269, 1290, 1291, 1293, 1286, 1288, 1251, 1267, 1274, 1285, 1225, 1282, 1297, 1286, 1244, 1205, 1259, 1286, 1226, 1260, 1264, 1292, 1275, 1292, 1255, 1261, 1294, 1218, 1267, 1254, 1296, 1232, 1297, 934, 1212, 1220, 1297, 1271, 1025, 1298, 1283, 1225, 1239, 1295, 
857, 1294, 1299, 1287, 1228, 1278, 1283, 1259, 1226, 1277, 1246, 1299, 1297, 1272, 1282, 1283, 1260, 1282, 1291, 1281, 1212, 1270, 1231, 1273, 1276, 1290, 1206, 973, 1276, 1298, 1271, 1287, 1265, 1288, 1122, 1267, 1266, 1255, 1278, 1144, 1296, 1274, 1295, 1276, 1281, 1284, 1266, 1264, 1277, 1279, 1273, 1289, 1288, 1262, 1186, 1292, 1208, 1041, 1291, 1280, 1292, 1268, 1299, 1293, 1284, 1255, 1143, 1143, 1291, 1280, 1269, 1276, 1291, 1288, 1280, 1297, 1290, 1248, 1269, 1153, 1120, 1216, 1155, 1290, 1283, 1276, 1268, 1225, 1291, 1294, 1105, 1285, 1281, 1004, 1275, 1275, 1297, 1290, 1278, 1218, 1261, 1298, 1291, 1281, 1279, 1289, 1266, 1181, 1220, 1180, 1149, 1250, 1124, 1298, 1256, 1272, 954, 1287, 1118, 1276, 1227, 1292, 1247, 1290, 1263, 1300, 1253, 1112, 1101, 1278, 1278, 1241, 1042, 1040, 1265, 1235, 1255, 1256, 1294, 1300, 1275, 1278, 1299, 1248, 894, 1295, 1288, 1173, 1269, 1278, 1283, 1199, 1295, 1261, 1294, 1283, 1286, 1159, 1276, 1273, 1297, 1293, 1259, 1211, 1287, 1262, 1287, 1251, 1252, 1253, 1296, 1278, 1291, 1256, 1277, 1019, 1203, 1256, 1286, 1298, 1139, 1296, 1253, 1297, 1271, 1290, 1253, 1224, 1266, 1254, 1259, 1283, 1274, 1228, 1288, 1232, 1239, 1289, 1280, 1273, 1275, 1293, 1137, 1281, 1294, 1297, 1293, 1286, 1285, 1243, 1294, 1292, 1278, 1290, 1018, 982, 1234, 1252, 1296, 1259, 1299, 1292, 1247, 1283, 1234, 1092, 1266, 1244, 1299, 1255, 1044, 1275, 1225, 1267, 1263, 1294, 1287, 1184, 1262, 1262, 1245, 1231, 1217, 1243, 1268, 915, 1287, 1229, 1235, 1232, 1276, 1236, 1271, 1288, 1273, 1237, 1294, 1272, 1255, 1247, 1259, 970, 1282, 1267, 1218, 1270, 1294, 1288, 1285, 1290, 1257, 1286, 1296, 1028, 1246, 1294, 1104, 1278, 1272, 1212, 1259, 1238, 1222, 1245, 1268, 1244, 1274, 1288, 1261, 1062, 1292, 1293, 1215, 1230, 1290, 1127, 1255, 1298, 1296, 1268, 1244, 1252, 1264, 1275, 1232, 1049, 1282, 1290, 1264, 1242, 1200, 1270, 1274, 1287, 1288, 1298, 1255, 1239, 1184, 1294, 1299, 1293, 1198, 1284, 1283, 1154, 1281, 1300, 1300, 1207, 1149, 1300, 1300, 1289, 1300, 1298, 1298, 1300, 1300, 1300, 1297, 1297, 1293, 1299, 1238, 1299, 1300, 1298, 1288, 1299, 1125, 1297, 1300, 1297, 1299, 1298, 1293, 1288, 1286, 1299, 1293, 1299, 1241, 1300, 1299, 1293, 1300, 1300, 1300, 1299, 1295, 1289, 1213, 1192, 1290, 1256, 1290, 1299, 1293, 1294, 1285, 1299, 1289, 1294, 1296, 1263, 1272, 1286, 1300, 1279, 1300, 1292, 1292, 1299, 1298, 1299, 1300, 1300, 1300, 1298, 1296, 1300, 1297, 1248]
    # target_imagenet1000_full=[[i]*(num_imagenet1000_full[i]) for i in range(1000)]
    for i in range(len(target_b)): x=x+target_b[i]
    # for i in range(len(target_imagenet1000_full)): x=x+target_imagenet1000_full[i]
    x=torch.Tensor(x)
    trainset = TensorDataset(data, x, train_transform)
    val_loader = TensorDataset(data, x, train_transform)
    trainset.nclass = 1000 # 10
    val_loader.nclass = 1000 # 10
    if args.load_memory:
        loader_real = ClassMemDataLoader(trainset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(trainset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    nclass = trainset.nclass
    nch, hs, ws = trainset[0][0].shape

    synset = Synthesizer(args, nclass, nch, hs, ws)
    # synset.init(loader_real, init_type="exist")
    synset.init(loader_real, init_type=args.init)
    
    torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data_init.pt'))
    path1=f"../results/dit-distillation/1k_minimax"
    path2=os.path.join(args.save_dir, f'data_init.pt')
    # os.system(f"python latent2images.py --model DiT-XL/2 --image-size 256 --ckpt /root/autodl-tmp/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt --save-dir {path1} --spec 1k --num-samples {args.ipc} --synpath {path2} --nclass 1000")
    # save_img(os.path.join(args.save_dir, 'init.png'),
    #          synset.data,
    #          unnormalize=False,
    #          dataname=args.dataset)

    # aug, aug_rand = diffaug(args)
    # save_img(os.path.join(args.save_dir, f'aug.png'),
    #          aug(synset.sample(0, max_size=args.batch_syn_max)[0]),
    #          unnormalize=True,
    #          dataname=args.dataset)

    # if not args.test:
    #     synset.test(args, val_loader, logger, bench=False)
    
    ### initial training
    # os.system(f"python train.py -d imagenet --imagenet_dir ../results/dit-distillation/1k_minimax /root/autodl-tmp/MinimaxDiffusion/data/imagenet/ -n vgg11 --nclass 1000 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k") # vgg11
    # os.system(f"python train.py -d imagenet --imagenet_dir ../results/dit-distillation/1k_minimax /root/autodl-tmp/MinimaxDiffusion/data/imagenet/ -n vit --nclass 1000 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k") # vit
    # os.system(f"python train.py -d imagenet --imagenet_dir ../results/dit-distillation/1k_minimax /root/autodl-tmp/MinimaxDiffusion/data/imagenet/ -n efficient --nclass 1000 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k") # efficient
    
    # os.system(f"python train.py -d imagenet --imagenet_dir ../results/dit-distillation/1k_500_1-10-minimax /root/autodl-tmp/MinimaxDiffusion/data/imagenet/     -n convnet --depth 6 --norm_type instance --nclass 500 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k_500_1")
    # os.system(f"python train.py -d imagenet --imagenet_dir ../results/dit-distillation/1k_500_1-10-minimax /root/autodl-tmp/MinimaxDiffusion/data/imagenet/     -n resnet_ap --nclass 500 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k_500_1")
    # os.system(f"python train.py -d imagenet --imagenet_dir {path1} /root/autodl-tmp/MinimaxDiffusion/data/imagenet/     -n resnet --depth 18 --norm_type instance --nclass 1000 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k")
    
    # args=test_imagenet(args,"../results/dit-distillation/imagenet-10-100-minimax")
    # ema = EMA(synset, 0.999)
    # ema.register()
    optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)

    ts = utils.TimeStamp(args.time)
    n_iter = args.niter * 100 // args.inner_loop
    it_log = n_iter // 50
    # it_test = [n_iter // 200, n_iter // 100, n_iter // 50, n_iter // 25, n_iter // 10, n_iter // 5, n_iter // 2, n_iter]

    logger(f"\nStart condensing with {args.match} matching for {args.niter} iteration")
    args.fix_iter = max(1, args.fix_iter)

    for it in range(args.niter):
        file_dir_set = args.model_path                             # path of checkpoints pretrained models 'pth.tar
        # print(file_dir_set)
        file_dir_set = os.listdir(file_dir_set[0])
        args.model_num = len(file_dir_set)

        path=f"/root/autodl-tmp/MinimaxDiffusion/pretrained_models/latent_1k/conv3in_cut_seed_2023_lr_0.1_aug_color_crop_cutout/checkpoint_best_0.pth.tar"
        current_state_dict = load_state(path)
        model = define_model(args, nclass)
        model.load_state_dict(current_state_dict)
        model_param = [device, nclass, path]
        # model_param = [device, nclass, path]
        xdirection, ydirection = setup_directions(args, model, model_param)
        w = get_weights(model)
        s = copy.deepcopy(model.state_dict())
        xcoordinates = random.uniform(args.vmax, args.vmin) * math.pow(-1, random.randint(1, 100))
        ycoordinates = random.uniform(args.vmax, args.vmin) * math.pow(-1, random.randint(1, 100))
        if args.dir_type == 'weights':
            set_weights(model, w, [xdirection, ydirection], [xcoordinates, ycoordinates])
        elif args.dir_type == 'states':
            set_states(model, s, [xdirection, ydirection], [xcoordinates, ycoordinates])
        
        model.train()
        model = model.to(device)
        optimizer_net = optim.SGD(model.parameters(),
                                  args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        
        criterion = nn.CrossEntropyLoss()
        loss_total = 0
        # synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)
        for ot in range(args.inner_loop):
            ts.set()

            for c in range(nclass):
                img, lab = loader_real.class_sample(c)
                img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
                ts.stamp("data")

                n = img.shape[0]
                img_aug = torch.cat([img, img_syn])
                ts.stamp("aug")

                loss = matchloss(args, img_aug[:n], img_aug[n:], lab, lab_syn, model,it)*0.1 ### DM
                loss_total += loss.item()
                ts.stamp("loss")

                optim_img.zero_grad()
                loss.backward()
                optim_img.step()
                # ema.update()
                ts.stamp("backward")

            if args.n_data > 0:
                for _ in range(args.net_epoch):
                    train_epoch(args,
                                loader_real,
                                model,
                                criterion,
                                optimizer_net,
                                _,
                                logger,
                                mixup=args.mixup)

            if (ot + 1) % 10 == 0:
                ts.flush()

        logger(
            f"{utils.get_time()} (Iter {it:3d}) loss: {loss_total / nclass / args.inner_loop:.1f}")

        if (it + 1) % args.val_interval == 0:###
        # if (it + 1) % 10 == 0:
            # save_img(os.path.join(args.save_dir, f'img{it + 1}.png'),
            #          synset.data,
            #          unnormalize=False,
            #          dataname=args.dataset)

            torch.save(
                [synset.data.detach().cpu(), synset.targets.cpu()],
                os.path.join(args.save_dir, f'data_last.pt'))
            print("img and data saved!")

            if not args.test:
                # ema.apply_shadow() ### ema
                # best_acc_e, last_acc_e = synset.test(args, val_loader, logger)
                # if best_acc_e <= last_acc_e:
                torch.save(
                    [synset.data.detach().cpu(), synset.targets.cpu()],
                    os.path.join(args.save_dir, f'data_best.pt'))
                print("best img and data updated!")
                # ema.restore() ### ema
                if args.factor>1:
                    data, target = torch.load(os.path.join(args.save_dir, f'data_best.pt'))
                    print("Load condensed data ", data.shape)
                    data, target = decode(args, data, target)
                    torch.save(
                    [data.detach().cpu(), target.cpu()],
                    os.path.join(args.save_dir, f'decode_data_best.pt'))
                    print("decode img and data updated!")
                    path1="../results/dit-distillation/imagenet-10-100-minimax-distill"
                    path2=os.path.join(args.save_dir, f'decode_data_best.pt')
                    os.system(f"python latent2images.py --model DiT-XL/2 --image-size 256 --ckpt ../logs/run-0/000-DiT-XL-2-minimax/checkpoints/0012000.pt --save-dir {path1} --spec woof --num-samples {args.ipc*args.factor**2} --synpath {path2}")
                    os.system(f"python train.py -d imagenet --imagenet_dir {path1} /root/autodl-tmp/MinimaxDiffusion/data/imagenet/     -n resnet_ap --nclass 10 --norm_type instance --ipc {args.ipc*args.factor**2} --tag test --slct_type random --spec woof")
                else:
                    path1=f"../results/dit-distillation/1k-distill"
                    path2=os.path.join(args.save_dir, f'data_best.pt')
                    os.system(f"python latent2images.py --model DiT-XL/2 --image-size 256 --ckpt /root/autodl-tmp/MinimaxDiffusion/pretrained_models/DiT-XL-2-256x256.pt --save-dir {path1} --spec 1k --num-samples {args.ipc} --synpath {path2} --nclass 1000")
                    
                    # os.system(f"python train.py -d imagenet --imagenet_dir {path1} /root/autodl-tmp/MinimaxDiffusion/data/imagenet/ -n vgg11 --nclass 1000 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k") # vgg11
                    os.system(f"python train.py -d imagenet --imagenet_dir {path1} /root/autodl-tmp/MinimaxDiffusion/data/imagenet/ -n vit --nclass 1000 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k") # vit
                    # os.system(f"python train.py -d imagenet --imagenet_dir {path1} /root/autodl-tmp/MinimaxDiffusion/data/imagenet/ -n efficient --nclass 1000 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k") # efficient

                    # os.system(f"python train.py -d imagenet --imagenet_dir {path1} /root/autodl-tmp/MinimaxDiffusion/data/imagenet/     -n convnet --depth 6 --nclass 500 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k_500_1")
                    # os.system(f"python train.py -d imagenet --imagenet_dir {path1} /root/autodl-tmp/MinimaxDiffusion/data/imagenet/     -n resnet_ap --nclass 500 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k_500_1")
                    # os.system(f"python train.py -d imagenet --imagenet_dir {path1} /root/autodl-tmp/MinimaxDiffusion/data/imagenet/     -n resnet --depth 18 --nclass 1000 --norm_type instance --ipc {args.ipc} --tag test --slct_type random --spec 1k")

                # args=test_imagenet(args,"../results/dit-distillation/imagenet-10-100-minimax-distill")

if __name__ == '__main__':
    import shutil
    from misc.utils import Logger
    from argument import args
    import torch.backends.cudnn as cudnn
    import json

    assert args.ipc > 0

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.save_dir)

    logger = Logger(args.save_dir,args.idx_1k)
    logger(f"Save dir: {args.save_dir}")
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    condense(args, logger)