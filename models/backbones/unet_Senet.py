from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import torch

from models.builder_QSQL import QSQLModel



import torch

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import json




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




class Unet(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_channels: Optional[List[int]] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        pretrained_path: Optional[str] = None,  
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        QSQL_model = QSQLModel(
            create_base_encoder(models.__dict__["resnet34"]),
            128,
            65536,
            0.999,
            0.07,
            True,
        )
        if pretrained_path:
            # 加载预训练模型的状态
            checkpoint = torch.load(pretrained_path)
            QSQL_model.load_state_dict(checkpoint["state_dict"], strict= False)
            print("Loading pretrain model")

        # 从模型中提取encoder_q的权重
        self.encoder.load_state_dict(QSQL_model.encoder_q.state_dict(), strict=False)

        
        if encoder_channels is None:
            encoder_channels = self.encoder.out_channels

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()
