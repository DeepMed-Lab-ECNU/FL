# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_, normal_, trunc_normal_
import matplotlib.pyplot as plt

from models.backbones.ops.modules import MSDeformAttn

import pandas as pd
import numpy as np

def count_frequency_samples(all_sampled_points):
    """
    Count the number of samples for each frequency band across all layers and queries.
    Args:
        all_sampled_points: A list of tensors, each of shape (B, num_query, num_points, 3),
                            where the last dimension represents (x, y, frequency_index).
    Returns:
        A tensor representing the count of samples for each frequency band.
    """
    # 初始化一个长度为60的零张量
    frequency_counts = torch.zeros(60, dtype=torch.int64)
    
    for sampled_points in all_sampled_points:
        # 遍历每个层的采样点
        for b in range(sampled_points.shape[0]):  # 遍历批次
            for q in range(sampled_points.shape[1]):  # 遍历查询
                # 获取每个采样点的频段索引
                freq_indices = sampled_points[b, q, :, -1].long()
                # 更新频段计数
                for index in freq_indices:
                    frequency_counts[index] += 1

    return frequency_counts

def visualize_frequency_distribution(frequency_counts):
    """
    Visualize the distribution of sample counts across frequency bands.
    Args:
        frequency_counts: A tensor representing the count of samples for each frequency band.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(torch.arange(len(frequency_counts)), frequency_counts.numpy(), color='skyblue')
    plt.xlabel('Frequency Band Index')
    plt.ylabel('Sample Count')
    plt.title('Distribution of Sampled Points Across Frequency Bands')
    plt.savefig('intensity_simple.png')  # 存储图形到文件
    
def cal_feature_redundance(feature, dim):
    feature = feature.squeeze(0).flatten(1).cpu().numpy().T  # N, C
    ft_pd = pd.DataFrame(feature)
    p = ft_pd.corr(method="pearson")
    p_np = np.abs(np.array(p))
    p_mean = p_np.sum() / (dim ** 2)

    return p_mean
    
class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4, num_cr_experts=1, num_br_experts=256):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.cr_expert = Parameter(torch.zeros(1, num_cr_experts, 2 * self.d_model))
        self.br_expert = Parameter(torch.zeros(1, num_br_experts, 2 * self.d_model))

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)
        
        
        
        self.cr_expert2 = Parameter(torch.zeros(1, num_cr_experts, 2 * self.d_model))
        self.br_expert2 = Parameter(torch.zeros(1, num_br_experts, 2 * self.d_model))

        decoder_layer2 = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder2 = DeformableTransformerDecoder(decoder_layer2, num_decoder_layers, return_intermediate_dec)

        self.level_embed2 = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points2 = nn.Linear(d_model, 2)
        
        

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        trunc_normal_(self.cr_expert, std=0.02)
        trunc_normal_(self.br_expert, std=0.02)
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def forward(self, src_flatten, src_spatial_shapes, src_level_start_index, src_valid_ratios, src_flatten_DownSampling, src_spatial_shapes_DownSampling, src_level_start_index_DownSampling, src_valid_ratios_DownSampling):
        B_DownSampling, C_DownSampling, N_DownSampling = src_flatten_DownSampling.shape[:3]
        # (B, C, N, 256) ---> (B*N, C, 256)
        band_flatten_DownSampling = src_flatten_DownSampling.permute(0, 2, 1, 3).contiguous().view(B_DownSampling * N_DownSampling, C_DownSampling, self.d_model)
        band_spatial_shapes_DownSampling = torch.tensor([1, C_DownSampling], device=src_flatten_DownSampling.device)[None, ...]
        band_valid_ratios_DownSampling = torch.ones([B_DownSampling * N_DownSampling, 1, 2], device=src_flatten_DownSampling.device)
        band_start_index_DownSampling = torch.tensor([0], device=src_flatten_DownSampling.device)

        # encoder
        memory = self.encoder(band_flatten_DownSampling, band_spatial_shapes_DownSampling, band_start_index_DownSampling, band_valid_ratios_DownSampling)  # (B*N, C, 256)
        # print(memory.shape)
        memory = memory.view(B_DownSampling, N_DownSampling * C_DownSampling, self.d_model)  # (B, N*C, 256)
        
        # prepare cr input for decoder
        query_embed = torch.cat((self.cr_expert, self.br_expert), dim=1)       
        query_embed, tgt = torch.split(query_embed, self.d_model, dim=2)  # 2x (1, num_query, 256)
        query_embed = query_embed.expand(B_DownSampling, -1, -1)  # (B, num_query, 256)
        tgt = tgt.expand(B_DownSampling, -1, -1)  # (B, num_query, 256)
        reference_points = self.reference_points(query_embed).sigmoid()  # (B, num_query, 2)

        # decoder  tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios
        hs, reference_points, all_sampled_points = self.decoder(tgt, reference_points, memory, src_spatial_shapes_DownSampling, src_level_start_index_DownSampling, src_valid_ratios_DownSampling, query_embed)
        
        
        B, C, N = src_flatten.shape[:3]
        band_flatten = src_flatten.permute(0, 2, 1, 3).contiguous().view(B * N, C, self.d_model)
        band_flatten = band_flatten.view(B, N * C, self.d_model)  # (B, N*C, 256)
        # prepare cr input for decoder
        query_embed2 = torch.cat((self.cr_expert2, self.br_expert2), dim=1)       
        query_embed2, tgt2 = torch.split(query_embed2, self.d_model, dim=2)  # 2x (1, num_query, 256)
        query_embed2 = query_embed2.expand(B, -1, -1)  # (B, num_query, 256)
        tgt2 = tgt2.expand(B, -1, -1)  # (B, num_query, 256)
        reference_points2 = self.reference_points2(query_embed2).sigmoid()  # (B, num_query, 2)

        # decoder  tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios
        hs2, reference_points2, all_sampled_points2 = self.decoder2(tgt2, reference_points2, band_flatten, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_embed2)
        
        
        
        # frequency_counts = count_frequency_samples(all_sampled_points)
        # visualize_frequency_distribution(frequency_counts)
        # visualize_sampled_points(all_sampled_points)
        

        return hs[:, 0], hs[:, 1:] , hs2[:, 1:]






class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgtx = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgtx)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgtx = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgtx)
        tgt = self.norm2(tgt)

        # cross attention
        tgtx, sampled_points = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgtx)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, sampled_points


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        all_sampled_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output, sampled_points = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            all_sampled_points.append(sampled_points)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points, all_sampled_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


