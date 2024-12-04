import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.backbones.deformable_transformer_parallel_DownSampling import DeformableTransformer
from models.backbones.unet_Senet import Unet


class DeformableTaskExpertS4RForMIM(nn.Module):
    def __init__(self, config):
        super(DeformableTaskExpertS4RForMIM, self).__init__()
        self.in_c = config.DATA.INPUT_CHANNEL
        self.mask_patch_size = config.DATA.MASK_PATCH_SIZE

        self.encoder, self.upsampler, self.rec_head = self.build_unet(config)
        self.alignment = self.build_alignment(config)
        self.alignment_DownSampling = self.build_alignment_DownSampling(config)
        self.ch_pos = Parameter(torch.zeros(config.DATA.INPUT_CHANNEL, config.MODEL.DIM))
        nn.init.trunc_normal_(self.ch_pos, std=0.02)

        self.decoder = self.build_decoder(config)

        self.mlp = nn.Sequential(
                nn.Linear(config.MODEL.DIM, 2 * config.MODEL.DIM),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(2 * config.MODEL.DIM, self.in_c - 1)
            )

    def build_unet(self, config):
        unet = Unet(
            encoder_name='resnet34',
            encoder_depth=config.MODEL.UNET_BLOCKS,
            encoder_weights="imagenet",
            encoder_channels=config.MODEL.UNET.ENCODER_CHANNELS,
            decoder_channels=config.MODEL.UNET.DECODER_CHANNELS,
            in_channels=3,
            classes=1,
            pretrained_path = config.MODEL.UNET.PRETRAIN_PATH,          
        )

        encoder = unet.encoder
        upsampler = unet.decoder
        head = unet.segmentation_head

        return encoder, upsampler, head

    def build_basic_layer(self, lvl, in_ch, out_ch):
        basic_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((8 * (4 - lvl), 8 * (4 - lvl))),
                nn.Conv2d(in_ch, out_ch, (1, 1)),
                nn.BatchNorm2d(out_ch),  
                nn.ReLU()
            )

        return basic_layer

    def build_alignment(self, config):
        alignment = nn.ModuleList([self.build_basic_layer(i, config.MODEL.FEATURE_CHANNELS[i],
                                                          config.MODEL.DIM) for i in range(4)])

        return alignment
    
    
    def build_basic_layer_DownSampling(self, lvl, in_ch, out_ch):
        basic_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((4 * (4 - lvl), 4 * (4 - lvl))),
                nn.Conv2d(in_ch, out_ch, (1, 1)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        return basic_layer

    def build_alignment_DownSampling(self, config):
        alignment = nn.ModuleList([self.build_basic_layer_DownSampling(i, config.MODEL.FEATURE_CHANNELS[i],
                                                          config.MODEL.DIM) for i in range(4)])

        return alignment
    
    
    

    def build_decoder(self, config):
        decoder = DeformableTransformer(
            d_model=config.MODEL.DIM,
            nhead=config.MODEL.HEAD,
            num_encoder_layers=config.MODEL.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.MODEL.NUM_DECODER_LAYERS,
            dim_feedforward=2*config.MODEL.DIM,
            dropout=config.MODEL.DROPOUT,
            activation="relu",
            return_intermediate_dec=config.MODEL.AUX_LOSS,
            num_feature_levels=4,
            dec_n_points=config.MODEL.NUM_DECODER_POINTS,
            enc_n_points=config.MODEL.NUM_ENCODER_POINTS,
            num_cr_experts=config.MODEL.NUM_CR_EXPERTS, 
            num_br_experts=config.MODEL.NUM_BR_EXPERTS)

        return decoder

    def patchify(self, imgs):
        """
        imgs: (N, C-1, H, W)
        x: (N*(C-1), L, patch_size**2)
        """

        B = imgs.shape[0] 
        p = self.mask_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(B, self.in_c, h, p, w, p))
        x = torch.einsum('nchpwq->nchwpq', x)
        x = x.reshape(shape=(B * self.in_c, h * w, p ** 2))
        return x

    def unpatchify(self, x):
        """
        x: (N*(C-1), L, patch_size**2)
        imgs: (N*(C-1), 1, H, W)
        """
        p = self.mask_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p))
        x = torch.einsum('nhwpq->nhpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask):
        B, C_ = x.shape[:2]
        extended_mask_idx = mask.unsqueeze(-1)  
        patches = self.patchify(x)  
        # 将 extended_mask_idx 广播到 patches 的维度
        patches[extended_mask_idx.expand(-1, -1, 256)] = 0
        
        x_ = self.unpatchify(patches)
        return x_

    def with_position_embedding(self, z, pos):
        pos_embed = self.ch_pos[pos] 
        z = z + pos_embed[:, :, None, :].expand(-1, -1, z.shape[2], -1)  

        return z
    
    def forward(self, x, mask):
        B, C_, H, W = x.shape
        x_ = self.random_masking(x, mask)  
        C_ = C_ // 3
        x_ = x_.view(B * C_, 3, H, W)
        feature_list = self.encoder(x_)
        # prepare input for encoder
        src_flatten = []
        spatial_shapes = []
        for lvl, sub_alignment in enumerate(self.alignment):
            src = sub_alignment(feature_list[-4:][lvl])  
            h, w = src.shape[-2:]
            spatial_shape = (h * C_, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1).view(B, C_, -1, 256)  
  
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  
        
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) 
        valid_ratios = torch.ones([B, spatial_shapes.shape[0], 2], device=src_flatten.device)  

        cr_experts, br_experts = self.decoder(src_flatten, spatial_shapes, level_start_index, valid_ratios)

        # coefficient regression
        beta = self.mlp(cr_experts.flatten(1))  

        # band regression
        H_ = W_ = int(br_experts.shape[1] ** 0.5)
        br_experts = br_experts.view(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous() 

        spatial_features = []
        for feature in feature_list[:-1]:
            c, h, w = feature.shape[-3:]
            feature = feature.view(B, -1, c, h, w)
            feature = torch.mean(feature, dim=1, keepdim=False)
            spatial_features.append(feature)
        spatial_features.append(br_experts)
        
        
        band_ft = self.upsampler(*spatial_features)
        band_ft = nn.functional.interpolate(
                    band_ft, size=(H, W),
                    mode="bilinear", align_corners=False
                )
        band_rec = self.rec_head(band_ft)

        return beta, band_rec
