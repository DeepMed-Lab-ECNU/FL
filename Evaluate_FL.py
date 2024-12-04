import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from local_utils.metrics import iou, dice
from Data_Generate import Data_Generate_Bile
from local_utils.seed_everything import seed_reproducer
from local_utils.misc import AverageMeter
from models.seg_models import DeformableTaskExpertS4R  
from local_utils.dice_bce_loss import Dice_BCE_Loss
from torch.utils.data import DataLoader, DistributedSampler


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
    net_type = args.net
    principal_bands_num = args.principal_bands_num

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

    test_files = dataset_dict['test']
    val_transformer = None
    

    test_images_path = [os.path.join(images_root_path, i) for i in test_files]
    test_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in test_files]

    # 加载数据集
    test_db = Data_Generate_Bile(test_images_path, test_masks_path, transform=val_transformer,
                                 principal_bands_num=principal_bands_num, cutting=cutting)
    test_sampler = DistributedSampler(test_db)
    test_loader = DataLoader(test_db, sampler=test_sampler, batch_size=batch, shuffle=False, num_workers=worker,
                             drop_last=False)



    if net_type == 'dual':
    # 定义配置参数
        config_dict = {
            "DATA": {
                "INPUT_CHANNEL": 60,  # 高光谱数据的光谱通道数
                "MASK_PATCH_SIZE": 16,  # 掩码补丁的大小
                "NUM_CLASS": 1,
            },
            "MODEL": {
                "DIM": 256,  # 模型中的维度，可能需要根据具体需求进行调整
                "UNET_BLOCKS": 5,  # Unet网络的深度或块的数量
                "UNET": {
                    "ENCODER_CHANNELS": [3, 64, 64, 128, 256, 256],  # Unet编码器的通道数
                    "DECODER_CHANNELS": [256, 128, 64, 32,16 ],  # Unet解码器的通道数
                    "in_channels": 60,  # 输入通道数应与高光谱数据的光谱通道数一致
                    "classes": 1,  # 根据任务需求调整
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeformableTaskExpertS4R(config).to(device)
    
    else:
            raise ValueError("Oops! That was no valid model. Try again...")

    model.load_state_dict(torch.load(args.pretrained_model)) 
    model.eval()
    test_losses = AverageMeter()
    labels, outs = [], []
    criterion = Dice_BCE_Loss()  
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_loader)):
            
            
            x1, label = sample
            x1 = x1[:, :, :, :256]
            label = label[:, :, :, :256]
            x1, label = x1.to(device), label.to(device)

            out, _ = model(x1)
            
            
            loss = criterion(out, label)
            test_losses.update(loss.item())
            out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
            outs.extend(out)
            labels.extend(label)

    outs, labels = np.array(outs), np.array(labels)
    outs = np.where(outs > 0.5, 1, 0)
    test_iou = np.array([iou(l, o) for l, o in zip(labels, outs)]).mean()
    test_dice = np.array([dice(l, o) for l, o in zip(labels, outs)]).mean()
    print(f'Test Loss: {test_losses.avg:.6f}, Test IoU: {test_iou:.6f}, Test Dice: {test_dice:.6f}')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_patch_size', type=int, default=16,
                        help='Mask patch size for DeformableTaskExpertS4RForMIM')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--root_path', '-r', type=str, default='./mdc_dataset/MDC')
    parser.add_argument('--dataset_hyper', '-dh', type=str, default='MHSI')
    parser.add_argument('--dataset_mask', '-dm', type=str, default='Mask')
    parser.add_argument('--dataset_divide', '-dd', type=str, default='train_val_test_DownSampling.json')

    parser.add_argument('--worker', '-nw', type=int,
                        default=4)
    parser.add_argument('--use_half', '-uh', action='store_true', default=False)
    parser.add_argument('--batch', '-b', type=int, default=8)

    parser.add_argument('--spatial_pretrain', '-sp', action='store_true', default=False)
    parser.add_argument('--lr', '-l', default=3e-5, type=float)
    parser.add_argument('--wd', '-w', default=1e-4, type=float)
    parser.add_argument('--rank', '-rank', type=int, default=4)
    parser.add_argument('--spectral_channels', '-spe_c', default=60, type=int)
    parser.add_argument('--principal_bands_num', '-pbn', default=-1, type=int)
    parser.add_argument('--use_aug', '-aug', action='store_true', default=True)
    parser.add_argument('--output', '-o', type=str, default='./bileseg-checkpoint')
    parser.add_argument('--experiment_name', '-name', type=str, default='Dual_MHSI')
    parser.add_argument('--decode_choice', '-de_c', default='unet', choices=['unet', 'fpn', 'deeplabv3plus'])
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--classes', '-c', type=int, default=1)
    parser.add_argument('--loss_function', '-lf', default='dicebce', choices=['dicebce', 'bce'])
    parser.add_argument('--hw', '-hw', type=int, default=[128, 128], nargs='+')
    parser.add_argument('--cutting', '-cut', default=-1, type=int)
    parser.add_argument('--scheduler', '-sc', default='cos', choices=['cos', 'warmup'])
    parser.add_argument('--net', '-n', default='dual', type=str, choices=['backbone', 'dual'])
    parser.add_argument('--backbone', '-backbone', default='resnet34', type=str)
    parser.add_argument('--attention_group', '-att_g', type=str, default='non', choices=['non', 'lowrank'])
    parser.add_argument('--pretrained_model', '-pm', type=str,default=None)
    args = parser.parse_args()
    main(args)
