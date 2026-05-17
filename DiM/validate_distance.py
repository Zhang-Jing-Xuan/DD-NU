# ipc=1 b=10
# ipc=10 b=25
# ipc=50 b=50
import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN
from gan_model import Generator, Discriminator
from utils import AverageMeter, accuracy, Normalize, Logger, rand_bbox
from augment import DiffAug
import shutil
from utils import get_strategy
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from data import Data
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


def load_data(args):
    '''Obtain data
    '''
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.data == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True,
                                   transform=transform_test)
        trainset.nclass = 10
    elif args.data == 'svhn':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197))
        ])
        trainset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                 split='train',
                                 download=True,
                                 transform=transform_train)
        testset = datasets.SVHN(os.path.join(args.data_dir, 'svhn'),
                                split='test',
                                download=True,
                                transform=transform_test)
        trainset.nclass = 10
    elif args.data == 'fashion':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])

        trainset = datasets.FashionMNIST(args.data_dir, train=True, download=True,
                                 transform=transform_train)
        testset = datasets.FashionMNIST(args.data_dir, train=False, download=True,
                                 transform=transform_train)
        trainset.nclass = 10
    elif args.data == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.131,), (0.308,))
        ])

        trainset = datasets.MNIST(args.data_dir, train=True, download=True,
                                 transform=transform_train)
        testset = datasets.MNIST(args.data_dir, train=False, download=True,
                                 transform=transform_train)
        trainset.nclass = 10
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    # testloader = MultiEpochsDataLoader(testset,
    #                                    batch_size=args.batch_size // 2,
    #                                    shuffle=False,
    #                                    persistent_workers=True,
    #                                    num_workers=4)
    return trainloader, trainset, testloader


def define_model(args, num_classes, e_model=None):
    '''Obtain model for training, validating and matching
    With no 'e_model' specified, it returns a random model
    '''
    if e_model:
        model = e_model
    else:
        model_pool = ['convnet', 'resnet10', 'resnet18',
                      'resnet10_ap', 'resnet18_ap']
        model = random.choice(model_pool)
        print('Random model: {}'.format(model))

    if args.data == 'mnist' or args.data == 'fashion':
        nch = 1
    else:
        nch = 3

    if model == 'convnet':
        return model, CN.ConvNet(num_classes, channel=nch)
    elif model == 'resnet10':
        return model, RN.ResNet(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18':
        return model, RN.ResNet(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34':
        return model, RN.ResNet(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50':
        return model, RN.ResNet(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101':
        return model, RN.ResNet(args.data, 101, num_classes, nch=nch)
    elif model == 'resnet10_ap':
        return model, RNAP.ResNetAP(args.data, 10, num_classes, nch=nch)
    elif model == 'resnet18_ap':
        return model, RNAP.ResNetAP(args.data, 18, num_classes, nch=nch)
    elif model == 'resnet34_ap':
        return model, RNAP.ResNetAP(args.data, 34, num_classes, nch=nch)
    elif model == 'resnet50_ap':
        return model, RNAP.ResNetAP(args.data, 50, num_classes, nch=nch)
    elif model == 'resnet101_ap':
        return model, RNAP.ResNetAP(args.data, 101, num_classes, nch=nch)
    elif model == 'densenet':
        return model, DN.densenet_cifar(num_classes)


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
    if args.data == 'cifar10':
        normalize = Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201), device='cuda')
    elif args.data == 'svhn':
        normalize = Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197), device='cuda')
    elif args.data == 'fashion':
        normalize = Normalize((0.286,), (0.353,), device='cuda')
    elif args.data == 'mnist':
        normalize = Normalize((0.131,), (0.308,), device='cuda')
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def test(args, model, testloader, criterion):
    '''Calculate accuracy
    '''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (img, lab) in enumerate(testloader):
        img = img.cuda()
        lab = lab.cuda()

        with torch.no_grad():
            output = model(img)
        loss = criterion(output, lab)
        acc1, acc5 = accuracy(output.data, lab, topk=(1, 5))
        losses.update(loss.item(), output.shape[0])
        top1.update(acc1.item(), output.shape[0])
        top5.update(acc5.item(), output.shape[0])

    return top1.avg, top5.avg, losses.avg

def get_init_images(args):
    def get_init_images(c,n):
    
        query_idxs= strategy_init.query(c,n)

        return images_all[query_idxs]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_pool = ['convnet', 'resnet10', 'resnet18',
                      'resnet10_ap', 'resnet18_ap']
    model_pool = ["convnet"]
    model_name, model = define_model(args, args.num_classes,"convnet")
    model = model.cuda()
    images_all = []
    labels_all = []
    images_all = [torch.unsqueeze(trainset[i][0], dim=0) for i in range(len(trainset))]
    labels_all = [trainset[i][1] for i in range(len(trainset))]
    images_all = torch.cat(images_all, dim=0).to("cuda")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cuda")
    dataset = Data(images_all, labels_all)
    trainloader_k = ClassMemDataLoader(trainset, batch_size=args.batch_size)
    nclass = trainloader_k.nclass
    query_list=torch.tensor(np.ones(shape=(nclass,args.batch_size)), dtype=torch.long, requires_grad=False, device=device)
    strategy_init = get_strategy('KMeansSampling')(dataset, model)
    args.fipc =1
    img_init=torch.zeros((nclass*args.fipc,3,32,32))
    for c in range(nclass):
        img_init[c*args.fipc:(c+1)*args.fipc] = get_init_images(c, args.fipc).detach().data
    torch.save(
                [img_init.detach().cpu()],
                os.path.join(args.logs_dir, f'init_data.pt'))
    for m in model_pool:
        
        model_name, Premodel = define_model(args, args.num_classes,m)
        Premodel = Premodel.cuda()
        Premodel.eval()
        head="/home/dd/DL/DataDistillation/Acc-DD/pretrained_models/cifar10/"
        tail="/checkpoint_best_0.pth.tar"
        if m=="convnet":
            path="conv3in_cut_seed_2023_lr_0.01_aug_color_crop_cutout"
        elif m=="resnet10":
            path="resnet10_cut_seed_2023_lr_0.01_aug_color_crop_cutout"
        elif m=="resnet18":
            path="resnet18_cut_seed_2023_lr_0.01_aug_color_crop_cutout"
        elif m=="resnet10_ap":
            path="resnet10ap_cut_seed_2023_lr_0.01_aug_color_crop_cutout"
        elif m=="resnet18_ap":
            path="resnet18ap_cut_seed_2023_lr_0.01_aug_color_crop_cutout"
        checkpoint = torch.load(
                head+path+tail
            )
        Premodel.load_state_dict(checkpoint)
        f_c = Premodel.get_feature(img_init.cuda(),0,3)
        torch.save(f_c,os.path.join(args.logs_dir, 'f_1_{}.pt'.format(m)))

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
 
    def update(self,decay):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
 
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
def validate(args, generator, testloader, criterion, aug_rand):
    '''Validate the generator performance
    '''
    all_best_top1 = []
    all_best_top5 = []
    model_name_P, Premodel = define_model(args, args.num_classes,"convnet")
    Premodel = Premodel.cuda()
    Premodel.eval()
    checkpoint = torch.load("/home/dd/DL/DataDistillation/Acc-DD/pretrained_models/cifar10/conv3in_cut_seed_2023_lr_0.01_aug_color_crop_cutout/checkpoint_best_0.pth.tar")
    Premodel.load_state_dict(checkpoint)
    for e_model in args.eval_model:
        print('Evaluating {}'.format(e_model))
        model_name,model = define_model(args, args.num_classes, e_model)
        model = model.cuda()
        model.train()
        ema = EMA(model, 0.99)
        ema.register()
        optim_model = torch.optim.SGD(model.parameters(), args.eval_lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)

        generator.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_top1 = 0.0
        best_top5 = 0.0

        # if args.batch_size == args.num_classes:
        #     lab_syn = torch.randperm(args.num_classes)
        # else:
        #     lab_syn = torch.randint(args.num_classes, (args.batch_size,))
        # noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        # lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        # lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
        # noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
        f_c = torch.load(os.path.join("/home/dd/DL/DataDistillation/DiM/logs/match_kp", 'f_1_{}.pt'.format(model_name)))
        img_syn_cache,label_cache=torch.Tensor([]),torch.Tensor([])
        img_syn_b_cache,label_b_cache=torch.Tensor([]),torch.Tensor([])
        distance_cache=torch.Tensor([])
        for epoch_idx in range(args.epochs_eval):
            for batch_idx in range(10 * args.ipc // args.batch_size):
                # obtain pseudo samples with the generator
                if args.batch_size == args.num_classes:
                    lab_syn = torch.randperm(args.num_classes)
                else:
                    lab_syn = torch.randint(args.num_classes, (args.batch_size,))
                noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
                lab_onehot = torch.zeros((args.batch_size, args.num_classes))
                lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
                noise[torch.arange(args.batch_size), :args.num_classes] = lab_onehot[torch.arange(args.batch_size)]
                

                noise = noise.cuda()
                lab_syn = lab_syn.cuda()

                with torch.no_grad():
                
                    img_syn = generator(noise)
                    img_syn = aug_rand((img_syn + 1.0) / 2.0)

                    distance = torch.Tensor([0.]).cuda()
                    # r_idx = [random.randint(args.fipc*lab_syn[idx],args.fipc*(lab_syn[idx]+1)-1) for idx in range(lab_syn.shape[0])]
                    for i in range(0,3):
                        f_syn = Premodel.get_feature(img_syn,0,3)[i]
                        # f_cc = f_c[i][r_idx].cuda()
                        f_cc = f_c[i][lab_syn].cuda()
                        distance = distance + ((f_syn-f_cc)**2).mean()
                        # res=((f_syn-f_cc)**2).view(args.batch_size,-1).mean(dim=1)
                        # if distance.shape[0]==0:
                        #     distance=res[torch.argsort(lab_syn)]
                        # else:
                        #     distance=distance+res[torch.argsort(lab_syn)]
                    # print(torch.max(distance))
                    # if distance_cache.shape[0]==0:
                    #     distance_cache=torch.max(distance).unsqueeze(0)
                        # img_syn_cache=img_syn
                        # label_cache=lab_syn
                    # else:
                    #     distance_cache=torch.cat([distance_cache,torch.max(distance).unsqueeze(0)])
                        # img_syn_cache=torch.cat([img_syn_cache,img_syn])
                        # label_cache=torch.cat([label_cache,lab_syn])
                # img_syn=img_syn_cache[torch.argmin(distance_cache)*args.batch_size:(torch.argmin(distance_cache)+1)*args.batch_size]
                # lab_syn=label_cache[torch.argmin(distance_cache)*args.batch_size:(torch.argmin(distance_cache)+1)*args.batch_size]
                # img_syn_cache,label_cache=torch.Tensor([]),torch.Tensor([])
                # img_syn_b_cache,label_b_cache=torch.Tensor([]),torch.Tensor([])
                # distance_cache=torch.Tensor([])
                # save synthetic images
                torch.save([img_syn.detach().cpu(), lab_syn.cpu()],os.path.join(args.logs_dir, f'data_best.pt'))

                if np.random.rand(1) < args.mix_p and args.mixup_net == 'cut':
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(len(img_syn)).cuda()

                    lab_syn_b = lab_syn[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img_syn.size(), lam)
                    img_syn[:, :, bbx1:bbx2, bby1:bby2] = img_syn[rand_index, :, bbx1:bbx2, bby1:bby2]
                    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_syn.size()[-1] * img_syn.size()[-2]))

                    output = model(img_syn)
                    loss = criterion(output, lab_syn) * ratio + criterion(output, lab_syn_b) * (1. - ratio)
                else:
                    output = model(img_syn)
                    loss = criterion(output, lab_syn)

                acc1, acc5 = accuracy(output.data, lab_syn, topk=(1, 5))

                losses.update(loss.item(), img_syn.shape[0])
                top1.update(acc1.item(), img_syn.shape[0])
                top5.update(acc5.item(), img_syn.shape[0])

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()

                ema.update(0.99)
                # ema.update(min(0.99,7.7*distance)) # 51.010
                # ema.update(min(0.999,7.6*distance)) # 73.76
            if (epoch_idx + 1) % args.test_interval == 0:
                ema.apply_shadow()
                test_top1, test_top5, test_loss = test(args, model, testloader, criterion)
                ema.restore()
                print('[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}'.format(epoch_idx + 1, test_top1, test_top5))
                if test_top1 > best_top1:
                    best_top1 = test_top1
                    best_top5 = test_top5

        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)

    return all_best_top1, all_best_top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipc', type=int, default=50)
    parser.add_argument('--fipc', type=int, default=10)
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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + args.tag
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + '/outputs'):
        os.makedirs(args.output_dir + '/outputs')

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args.logs_dir = args.logs_dir + args.tag
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    sys.stdout = Logger(os.path.join(args.logs_dir, 'logs.txt'))

    os.makedirs(args.logs_dir, exist_ok=True)
    cur_file = os.path.join(os.getcwd(), __file__)
    shutil.copy(cur_file, args.logs_dir)


    print(args)

    trainloader, trainset, testloader = load_data(args)
    # get_init_images(args)
    generator = Generator(args).cuda()
    generator.load_state_dict(torch.load(args.pretrain_weight)['generator'])

    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)

    top1s, top5s = validate(args, generator, testloader, criterion, aug_rand)
    for e_idx, e_model in enumerate(args.eval_model):
        print('Evaluation for {}: Top1: {:.3f}, Top5: {:.3f}'.format(e_model, top1s[e_idx], top5s[e_idx]))
