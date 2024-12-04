import torch
from torch import nn

import segmentation_models_pytorch as smp
from models.grp_tes4r import DeformableTaskExpertS4RForMIM
from models.builder_QSQL import QSQLModel
import torch.nn.functional as F

import torchvision.models as models



def create_base_encoder(pretrained_model):
    def _base_encoder(num_classes):
        model = pretrained_model(pretrained=False)
        in_features = model.fc.in_features  
        model.fc = nn.Linear(in_features, num_classes)
        return model
    return _base_encoder

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SELayer, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.LayerNorm(in_channels),  
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Global average pooling
        x_se = self.global_avg_pooling(x)
        x_se = x_se.view(x_se.size(0), -1)

        # Fully-connected layers
        x_se = self.fc(x_se)

        x_se = x_se.view(x.size(0), x.size(1), 1, 1)

        # Element-wise multiplication with input x
        x = x * x_se

        return x


class DeformableTaskExpertS4R(nn.Module):
    def __init__(self, config):
        super(DeformableTaskExpertS4R, self).__init__()
        self.mode = config.MODEL.MODE
        self.ssl_model = DeformableTaskExpertS4RForMIM(config)

        if config.MODEL.PRETRAIN:
            print("Load pretrain model")
            self.ssl_model.load_state_dict(torch.load('./bileseg-checkpoint/DFS3R_MHSI/epoch28_val_loss0.0519.pth'))
        self.encoder = self.ssl_model.encoder

        self.alignment = self.ssl_model.alignment
        self.alignment_DownSampling = self.ssl_model.alignment_DownSampling
        self.ch_pos = self.ssl_model.ch_pos
        self.decoder = self.ssl_model.decoder
        self.selayer = SELayer(config.DATA.INPUT_CHANNEL, 7)
        
        Pt_model = QSQLModel(
            create_base_encoder(models.__dict__["resnet34"]),
            128,
            65536,
            0.999,
            0.07,
            True,
        )

        checkpoint = torch.load(config.MODEL.UNET.PRETRAIN_PATH)
        Pt_model.load_state_dict(checkpoint["state_dict"], strict= False)
        print("Loading pretrain model")
        self.selayer.load_state_dict(Pt_model.selayer.state_dict(), strict=True)     

        # 为不同的维度定义层归一化
        self.ln1 = nn.LayerNorm([64, 64, 64])
        self.ln2 = nn.LayerNorm([128, 32, 32])
        self.ln3 = nn.LayerNorm([256, 16, 16])
        self.ln4 = nn.LayerNorm([512, 8, 8])
        
        # 初始化权重，大小为通道数 256
        self.weight1 = nn.Parameter(torch.ones(256))
        self.weight2 = nn.Parameter(torch.ones(256))
        
        if self.mode == 'classification':
            self.mlp = nn.Sequential(
                    nn.Linear(config.MODEL.DIM, 2 * config.MODEL.DIM),
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(2 * config.MODEL.DIM, config.DATA.NUM_CLASS)
                )

        elif self.mode == 'segmentation':
            self.upsampler = self.ssl_model.upsampler
            self.seg_head = nn.Conv2d(config.MODEL.UNET.DECODER_CHANNELS[-1], 
                                      config.DATA.NUM_CLASS,
                                      kernel_size=3, padding=3 // 2)
    
    def with_position_embedding(self, z):
        pos_embed = self.ch_pos.unsqueeze(0).expand(z.shape[0], -1, -1)  
        z = z + pos_embed[:, :, None, :].expand(-1, -1, z.shape[2], -1)  

        return z

    def forward(self, x):
        
        x = self.selayer(x)
        
        B, C, H, W = x.shape
        C_ = C // 3
        x = x.view(B * C_, 3, H, W)
        feature_list = self.encoder(x)

        # prepare input for encoder
        src_flatten = []
        spatial_shapes = []
        
        src_flatten_DownSampling = []
        spatial_shapes_DownSampling = []
        
        for lvl, sub_alignment in enumerate(self.alignment):
            
            abcd = feature_list[-4:][lvl]
            # 应用LN
            if lvl == 0:
                abcd = self.ln1(abcd)
            elif lvl == 1:
                abcd = self.ln2(abcd)
            elif lvl == 2:
                abcd = self.ln3(abcd)
            elif lvl == 3:
                abcd = self.ln4(abcd)
            src = sub_alignment(abcd)
            
            h, w = src.shape[-2:]
            spatial_shape = (h * C_, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            src_flatten.append(src)
            
            src_DownSampling = self.alignment_DownSampling[lvl](abcd)
            h_DownSampling, w_DownSampling = src_DownSampling.shape[-2:]
            spatial_shape_DownSampling = (h_DownSampling * C_, w_DownSampling)
            spatial_shapes_DownSampling.append(spatial_shape_DownSampling)
            src_DownSampling = src_DownSampling.flatten(2).transpose(1, 2)
            src_flatten_DownSampling.append(src_DownSampling)
            
            
        src_flatten = torch.cat(src_flatten, 1).view(B, C_, -1, 256) 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device) 
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))  
        valid_ratios = torch.ones([B, spatial_shapes.shape[0], 2], device=src_flatten.device) 
        
        
        src_flatten_DownSampling = torch.cat(src_flatten_DownSampling, 1).view(B, C_, -1, 256)  
        spatial_shapes_DownSampling = torch.as_tensor(spatial_shapes_DownSampling, dtype=torch.long, device=src_flatten_DownSampling.device) 
        level_start_index_DownSampling = torch.cat((spatial_shapes_DownSampling.new_zeros((1, )), spatial_shapes_DownSampling.prod(1).cumsum(0)[:-1])) 
        valid_ratios_DownSampling = torch.ones([B, spatial_shapes_DownSampling.shape[0], 2], device=src_flatten_DownSampling.device)  
        

        cr_experts, br_experts, br_experts2 = self.decoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, src_flatten_DownSampling, spatial_shapes_DownSampling, level_start_index_DownSampling, valid_ratios_DownSampling)

        # classification
        if self.mode == 'classification':
            pred = self.mlp(cr_experts.flatten(1)) 

        # band regression
        elif self.mode == 'segmentation':
            
            weight1 = self.weight1.view(1, -1, 1, 1)
            weight2 = self.weight2.view(1, -1, 1, 1)
            
            
            H_ = W_ = int(br_experts.shape[1] ** 0.5)
            br_experts = br_experts.view(B, H_, W_, -1).permute(0, 3, 1, 2).contiguous()  
            
            H_2 = W_2 = int(br_experts2.shape[1] ** 0.5)
            br_experts2 = br_experts2.view(B, H_2, W_2, -1).permute(0, 3, 1, 2).contiguous()  
            
            spatial_features = []
            for feature in feature_list[:-1]:
                c, h, w = feature.shape[-3:]
                feature = feature.view(B, -1, c, h, w)
                feature = torch.mean(feature, dim=1, keepdim=False)
                spatial_features.append(feature)
            result_br_experts = weight1 * br_experts + weight2 * br_experts2
            spatial_features.append(result_br_experts)

            ft = self.upsampler(*spatial_features) 
            pred = nn.functional.interpolate(
                    ft, size=(H, W),
                    mode="bilinear", align_corners=False
                )
            pred = self.seg_head(pred) 

        return pred, result_br_experts