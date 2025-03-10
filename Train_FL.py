import argparse
import json
import os
import signal
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from segmentation_models_pytorch.utils.losses import DiceLoss

from local_utils.tools import save_dict
from local_utils.seed_everything import seed_reproducer
from local_utils.misc import AverageMeter
from local_utils.dice_bce_loss import Dice_BCE_Loss
from local_utils.metrics import iou, dice
from Data_Generate import Data_Generate_Bile
from argument import Transform
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.seg_models import DeformableTaskExpertS4R 

def reduce_tensor(tensor):
    rt = torch.tensor(tensor).cuda()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= torch.cuda.device_count()
    return rt.cpu().numpy()



# 定义 Config 类
class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    seed_reproducer(args.seed)

    root_path = args.root_path
    dataset_hyper = args.dataset_hyper
    dataset_mask = args.dataset_mask
    dataset_divide = args.dataset_divide
    batch = args.batch

    lr = args.lr
    wd = args.wd
    use_aug = args.use_aug
    experiment_name = args.experiment_name
    output_path = args.output
    epochs = args.epochs
    use_half = args.use_half

    scheduler_type = args.scheduler
    net_type = args.net
    principal_bands_num = args.principal_bands_num

    lf = args.loss_function
    worker = args.worker
    cutting = args.cutting
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group(backend='nccl')

    images_root_path = os.path.join(root_path, dataset_hyper)
    mask_root_path = os.path.join(root_path, dataset_mask)
    dataset_json = os.path.join(root_path, dataset_divide)
    with open(dataset_json, 'r') as load_f:
        dataset_dict = json.load(load_f)

    train_files = dataset_dict['train']
    val_files = dataset_dict['val']
    test_files = dataset_dict['test']
    transform = Transform(Rotate_ratio=0.2, Flip_ratio=0.2, BrightContrast_ratio=0.1, noise_ratio=0.1)
    val_transformer = None

    if local_rank == 0:
        print(f'the number of trainfiles is {len(train_files)}')
        print(f'the number of valfiles is {len(val_files)}')
        print(f'the number of testfiles is {len(test_files)}')

    train_images_path = [os.path.join(images_root_path, i) for i in train_files]
    train_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in train_files]
    val_images_path = [os.path.join(images_root_path, i) for i in val_files]
    val_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in val_files]
    test_images_path = [os.path.join(images_root_path, i) for i in test_files]
    test_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in test_files]

    train_db = Data_Generate_Bile(train_images_path, train_masks_path, transform=transform,
                                  principal_bands_num=principal_bands_num, cutting=cutting)
    train_sampler = DistributedSampler(train_db)
    train_loader = DataLoader(train_db, sampler=train_sampler, batch_size=batch, num_workers=worker, drop_last=True)

    val_db = Data_Generate_Bile(val_images_path, val_masks_path, transform=val_transformer,
                                principal_bands_num=principal_bands_num, cutting=cutting)
    val_sampler = DistributedSampler(val_db)
    val_loader = DataLoader(val_db, sampler=val_sampler, batch_size=batch, shuffle=False, num_workers=worker,
                            drop_last=False)
    #
    test_db = Data_Generate_Bile(test_images_path, test_masks_path, transform=val_transformer,
                                 principal_bands_num=principal_bands_num, cutting=cutting)
    test_sampler = DistributedSampler(test_db)
    test_loader = DataLoader(test_db, sampler=test_sampler, batch_size=batch, shuffle=False, num_workers=worker,
                             drop_last=False)

    if local_rank == 0:
        if os.path.exists(f'{output_path}/{experiment_name}') == False:
            os.mkdir(f'{output_path}/{experiment_name}')
        save_dict(os.path.join(f'{output_path}/{experiment_name}', 'args.csv'), args.__dict__)


    if net_type == 'dual':
        
        config_dict = {
            "DATA": {
                "INPUT_CHANNEL": 60,  # 高光谱数据的光谱通道数
                "MASK_PATCH_SIZE": 16,  # 掩码补丁的大小
                "NUM_CLASS": 1,
            },
            "MODEL": {
                "DIM": 256,  # 模型中的维度
                "UNET_BLOCKS": 5,  
                "UNET": {
                    "ENCODER_CHANNELS": [3, 64, 64, 128, 256, 256],  # Unet编码器的通道数
                    "DECODER_CHANNELS": [256, 128, 64, 32,16 ],  # Unet解码器的通道数
                    "in_channels": 60,  # 输入通道数应与高光谱数据的光谱通道数一致
                    "classes": 1, 
                    "PRETRAIN_PATH": ""
                },
                "FEATURE_CHANNELS": [64, 128, 256, 512],  # 特征通道数，用于构建对齐模块
                "HEAD": 8,  # 解码器的头数
                "NUM_ENCODER_LAYERS": 2,  # 解码器的编码器层数
                "NUM_DECODER_LAYERS": 4,  # 解码器的解码器层数
                "DROPOUT": 0.1,  # Dropout比率
                "AUX_LOSS": False,  # 是否返回中间解码器层的输出
                "NUM_DECODER_POINTS": 8 * 20,  # 解码器点的数量
                "NUM_ENCODER_POINTS": 8,  # 编码器点的数量
                "NUM_CR_EXPERTS": 1,  # CR专家的数量
                "NUM_BR_EXPERTS": 64,  # BR专家的数量
                "MODE": 'segmentation',
                "PRETRAIN": False,
            }
        }

        

        config = Config(config_dict)

        # 初始化模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeformableTaskExpertS4R(config).to(device)
    else:
            raise ValueError("Oops! That was no valid model. Try again...")


    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    if scheduler_type == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    elif scheduler_type == 'warmup':
        scheduler = CosineAnnealingLR(optimizer, t_initial=args.epochs, warmup_t=9, warmup_prefix=True)
    else:
        raise ValueError("Oops! That was no valid scheduler_type.Try again...")

    if lf == 'dice':
        criterion = DiceLoss()
    elif lf == 'dicebce':
        criterion = Dice_BCE_Loss(bce_weight=0.5, dice_weight=0.5)
    else:
        raise ValueError("Oops! That was no valid lossfunction.Try again...")



    history = {'epoch': [], 'LR': [], 'train_loss': [], 'val_loss': [], 'val_iou': [],
               'val_dice': [], 'test_iou': [], 'test_dice': [], }
    
    stop_training = False

    def sigint_handler(signal, frame):
        print("Ctrl+c caught, stopping the training and saving the model...")
        nonlocal stop_training
        stop_training = True
        history_pd = pd.DataFrame(history)
        history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', 'log.csv'), index=False)

    signal.signal(signal.SIGINT, sigint_handler)

    if use_half:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    best_val = 0.0  

    # 训练循环
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        train_losses = AverageMeter()
        val_losses = AverageMeter()
        if local_rank == 0:
            print('now start train ..')
            print('epoch {}/{}, LR:{}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
        train_losses.reset()
        val_losses.reset()

        # 训练步骤
        model.train()

        try:
            for idx, sample in enumerate(tqdm(train_loader)):
                if stop_training:
                    break
                x1, label = sample
                x1 = x1[:, :, :, :256]
                label = label[:, :, :, :256]           
                x1, label = x1.to(device), label.to(device)
                out , br_experts = model(x1)              
                loss = criterion(out, label)

                if use_half:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        optimizer.zero_grad()
                        scaled_loss.backward()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()
                train_losses.update(loss.item())

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, please reduce batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return
            else:
                raise e

        print('now start validation ...')
        model.eval()

        labels, outs = [], []
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(val_loader)):
                if stop_training:
                    break
                x1, label = sample
                x1 = x1[:, :, :, :256]
                label = label[:, :, :, :256]
                x1, label = x1.to(device), label.to(device)

                out , br_experts = model(x1)
                loss = criterion(out, label)

                val_losses.update(loss.item())
                out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
                outs.extend(out)

                labels.extend(label)
        outs, labels = np.array(outs), np.array(labels)
        outs = np.where(outs > 0.5, 1, 0)
        val_iou = np.array([iou(l, o) for l, o in zip(labels, outs)]).mean()
        val_dice = np.array([dice(l, o) for l, o in zip(labels, outs)]).mean()

        # 计算平均验证损失
        val_loss_avg = val_losses.avg
        
        
        print('now start test ...')
        model.eval()

        labels, outs = [], []
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(test_loader)):
                if stop_training:
                    break
                x1, label = sample
                x1 = x1[:, :, :, :256]
                label = label[:, :, :, :256]
                x1, label = x1.to(device), label.to(device)
                out , br_experts = model(x1)
                out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
                outs.extend(out)

                labels.extend(label)
        outs, labels = np.array(outs), np.array(labels)
        outs = np.where(outs > 0.5, 1, 0)
        test_iou = np.array([iou(l, o) for l, o in zip(labels, outs)]).mean()
        test_dice = np.array([dice(l, o) for l, o in zip(labels, outs)]).mean()
        
        
        
        print('epoch {}/{}\t LR:{}\t train loss:{}\t val loss:{}\t val_dice:{}\t test_dice:{}\n\n' \
            .format(epoch + 1, epochs, optimizer.param_groups[0]['lr'], train_losses.avg, val_loss_avg, val_dice, test_dice))
        

        # 更新历史记录   
        history['train_loss'].append(reduce_tensor(train_losses.avg))
        history['val_loss'].append(reduce_tensor(val_losses.avg))

        history['val_iou'].append(reduce_tensor(val_iou))
        history['val_dice'].append(reduce_tensor(val_dice))
        history['test_iou'].append(reduce_tensor(test_iou))
        history['test_dice'].append(reduce_tensor(test_dice))
        history['epoch'].append(epoch + 1)
        history['LR'].append(optimizer.param_groups[0]['lr'])

        # 更新学习率调度器
        if scheduler_type == 'warmup':
            scheduler.step(epoch)
        else:
            scheduler.step()

        # 检查是否提前停止训练
        if stop_training:
            torch.save(model.state_dict(),
                       os.path.join(f'{output_path}/{experiment_name}', 'final.pth'))
            break

        
        if best_val <= reduce_tensor(test_dice):
            if local_rank==0:
                if epoch > 0 and os.path.exists(save_path):
                    os.remove(save_path)

            best_val = reduce_tensor(test_dice)
            torch.save(model.state_dict(),
                    os.path.join(f'{args.output}/{args.experiment_name}',
                                    f'best_epoch{epoch}_dice{best_val:.4f}.pth'))
            save_path = os.path.join(f'{args.output}/{args.experiment_name}',
                                    f'best_epoch{epoch}_dice{best_val:.4f}.pth')
        


        # 如果是主进程，则保存训练历史
        if local_rank == 0:
            history_pd = pd.DataFrame(history)
            history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', f'TrainLog.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_patch_size', type=int, default=16, help='Mask patch size ')
    parser.add_argument("--seed", type=int, default=2333)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--root_path', '-r', type=str, default='./mdc_dataset/MDC')
    parser.add_argument('--dataset_hyper', '-dh', type=str, default='MHSI')
    parser.add_argument('--dataset_mask', '-dm', type=str, default='Mask')
    parser.add_argument('--dataset_divide', '-dd', type=str, default='train_val_test_DownSampling.json')

    parser.add_argument('--worker', '-nw', type=int,
                        default=4)
    parser.add_argument('--use_half', '-uh', action='store_true', default=False)
    parser.add_argument('--batch', '-b', type=int, default=8)

    parser.add_argument('--spatial_pretrain', '-sp', action='store_true', default=False)
    parser.add_argument('--lr', '-l', default=4e-4, type=float)
    parser.add_argument('--wd', '-w', default=1e-4, type=float)
    parser.add_argument('--spectral_hidden_feature', '-shf', default=64, type=int)

    parser.add_argument('--rank', '-rank', type=int, default=4)
    parser.add_argument('--spectral_channels', '-spe_c', default=60, type=int)
    parser.add_argument('--principal_bands_num', '-pbn', default=-1, type=int)
    parser.add_argument('--conver_bn2gn', '-b2g', action='store_true', default=False)
    parser.add_argument('--use_aug', '-aug', action='store_true', default=True)
    parser.add_argument('--output', '-o', type=str, default='./bileseg-checkpoint')
    parser.add_argument('--experiment_name', '-name', type=str, default='Dual_MHSI')
    parser.add_argument('--decode_choice', '-de_c', default='unet', choices=['unet', 'fpn', 'deeplabv3plus'])
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--classes', '-c', type=int, default=1)
    parser.add_argument('--bands_group', '-b_group', type=int, default=1)
    parser.add_argument('--link_position', '-link_p', type=int, default=[0, 0, 0, 1, 0, 1], nargs='+')
    parser.add_argument('--loss_function', '-lf', default='dicebce', choices=['dicebce', 'bce'])
    parser.add_argument('--spe_kernel_size', '-sks', type=int, default=1)#, nargs='+')

    parser.add_argument('--hw', '-hw', type=int, default=[128, 128], nargs='+')
    parser.add_argument('--spa_reduction', '-sdr', type=int, default=[4, 4], nargs='+')
    parser.add_argument('--cutting', '-cut', default=-1, type=int)
    parser.add_argument('--merge_spe_downsample', '-msd', type=int, default=[2, 1], nargs='+')
    parser.add_argument('--scheduler', '-sc', default='cos', choices=['cos', 'warmup'])
    parser.add_argument('--net', '-n', default='dual', type=str, choices=['backbone', 'dual'])
    parser.add_argument('--backbone', '-backbone', default='resnet34', type=str)
    parser.add_argument('--attention_group', '-att_g', type=str, default='non', choices=['non', 'lowrank'])
    args = parser.parse_args()

    main(args)
