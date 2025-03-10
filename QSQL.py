#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import random
import time
import warnings

from models.builder_QSQL import QSQLModel
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from Data_Generate_QSQL import Data_Generate_Bile
from torch.utils.data import DataLoader
import json
from argument_QSQL import Transform

class SpectralJitter:
    def __init__(self, max_change=0.05):
        self.max_change = max_change

    def __call__(self, x):
        # x: 高光谱图像，假设其形状为 (Channels, Height, Width)
        noise = torch.randn_like(x) * self.max_change
        return x + noise



class NoiseInjection:
    def __init__(self, noise_level=0.01):
        self.noise_level = noise_level

    def __call__(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

def reduce_tensor(tensor):
    rt = torch.tensor(tensor).cuda()
    # dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= torch.cuda.device_count()
    return rt.cpu().numpy()


def create_base_encoder(pretrained_model):
    def _base_encoder(num_classes):
        # 加载预训练模型，但不包括最后的全连接层
        model = pretrained_model(pretrained=True)
        # 替换模型的最后全连接层以适应新的维度
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    return _base_encoder

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet34",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet34)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=300, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[100, 200, 250],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
# parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    default=False,
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

parser.add_argument(
    "--QSQLModel-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--QSQLModel-k",
    default=65536,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--QSQLModel-m",
    default=0.999,
    type=float,
    help="momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--QSQLModel-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

parser.add_argument("--mlp", action="store_true", help="use mlp head")

parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format(args.arch))
    
    base_encoder = models.__dict__[args.arch](pretrained=True)
    # 创建模型
    model = QSQLModel(
        create_base_encoder(models.__dict__[args.arch]),
        args.QSQLModel_dim,
        args.QSQLModel_k,
        args.QSQLModel_m,
        args.QSQLModel_t,
        args.mlp,
    )
    
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
        pass

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    criterion2 = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # 配置参数
    root_path = '/root/autodl-tmp/Dual-Stream-MHSI-main/MDC_dataset'  # 数据根目录
    dataset_hyper = 'MHSI'  # 高光谱数据集目录
    dataset_mask = 'Mask'  # 掩码数据集目录
    dataset_divide = 'train_val_test_DownSampling.json'  # 数据集划分文件
    batch = args.batch_size  # 批大小
    worker = 4  # 工作线程数
    principal_bands_num = -1  # 主要波段数
    cutting = -1  # 裁剪设置
    use_aug = False  # 是否使用数据增强

    # 数据路径
    images_root_path = os.path.join(root_path, dataset_hyper)
    mask_root_path = os.path.join(root_path, dataset_mask)
    dataset_json = os.path.join(root_path, dataset_divide)

    # 读取数据集划分
    with open(dataset_json, 'r') as load_f:
        dataset_dict = json.load(load_f)
    train_files = dataset_dict['train']

    # 构建训练数据路径
    train_images_path = [os.path.join(images_root_path, i) for i in train_files]
    train_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in train_files]

    # 数据转换
    transform = Transform(
    Rotate_ratio=0.2, 
    Flip_ratio=0.2, 
    BrightContrast_ratio=0.1, 
    noise_ratio=0.1
    )

    train_db = Data_Generate_Bile(train_images_path, train_masks_path, transform=transform,
                                principal_bands_num=principal_bands_num, cutting=cutting)
    train_loader = DataLoader(train_db, batch_size=batch, num_workers=worker, drop_last=True)
    filename = "checkpoint.pth.tar"
    best_loss = 99999
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        epoch_loss = train(train_loader, model, criterion, criterion2, optimizer, epoch, args)

        if best_loss >= reduce_tensor(epoch_loss):
            if epoch > 0:
                 delete_checkpoint(filename)
            best_loss = reduce_tensor(epoch_loss)
            
            delete_checkpoint(filename)
            filename = f"checkpoint_{epoch:04d}_loss_{epoch_loss:.4f}.pth.tar"
            save_checkpoint({
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, filename=filename)
            

def train(train_loader, model, criterion, criterion2, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    
    
    
    
    for i, (images, indices) in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        actual_batch_size = images.size(0)
        if actual_batch_size * 20 < args.batch_size * 20:
            continue
        
        images = images[:, :, :, :256]
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            indices = indices.cuda(args.gpu, non_blocking=True) 

        # 将图像和对应的索引传递给模型
        total_loss = model(images, indices)

        losses.update(total_loss.item(), images.size(0))
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    root_path = './Pretrain_models_QSQL/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    final_name = os.path.join(root_path, filename)
    torch.save(state, final_name)
    
def delete_checkpoint(filename="checkpoint.pth.tar"):
    root_path = './Pretrain_models_QSQL/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    final_name = os.path.join(root_path, filename)
    if os.path.exists(final_name):
        os.remove(final_name)
    


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
