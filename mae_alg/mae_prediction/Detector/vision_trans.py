# ---------------------------------------------------
# References:
#   - torchvision source code
#   - https://github.com/facebookresearch/mae
#   - https://github.com/ViTAE-Transformer/ViTDet
# ---------------------------------------------------

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import timm.models.vision_transformer

from Detector.feature_pyramid_network import *
from Detector.models_vit import *
from Detector.vit import Norm2d


class VitDetAugmented(nn.Module):
    '''
    Base class to build the FeaturePyramidNetwork using ViT backbone
    '''

    def __init__(self, backbone, embed_dim=768, out_dim=256):
        super().__init__()
        self.backbone = backbone
        self.fpn1 = nn.Sequential(
            # Performs operations similar to DeConvolution: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
            # increases input size
            nn.ConvTranspose2d(
                embed_dim, embed_dim, kernel_size=2, stride=2),
            Norm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fpn_head = FeaturePyramidNetwork(
            in_channels_list=[768, 768, 768, 768],
            out_channels=256,
            norm_layer=Norm2d,
            extra_blocks=LastLevelMaxPool()
        )

        self.out_channels = out_dim

    def forward_features(self, x):
        '''
        Passes input through backbone and through 4 blocks to be fed as 
        input features into FeaturePyramidNetwork

        Args:
            x:  Batch of images to be fed into backbone and FeaturePyramidNetwork
                x should have shape (B, 3, 512, 512) as ViT model is expecting color images 
                of size 512 X 512
        Returns:
            features:   Dict of 5 features with sizes:
                            1: 2, 256, 128, 128
                            2: 2, 256, 64, 64
                            3: 2, 256, 32, 32
                            4: 2, 256, 16, 16
                            5: 2, 256, 8, 8
                        To used as FasterRCNN's Region Proposal Network
                        Number of values in dict must match number of features_names in region_of_interest
        '''
        # Inupt image batches are passed to ViT model with shape 2, 3, 512, 512
        # Output of backbone processing has shape 2, 768, 32, 32
        x = self.backbone(x)

        features = OrderedDict()
        archs = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]

        iterator = 0
        while (iterator < len(archs)):
            features[iterator] = archs[iterator](x)
            iterator += 1

        # Pass the dictionary of features created to the feature pyramid network
        features = self.fpn_head(features)

        # Return the features returned by the feature pyramid network
        return features

    def forward(self, x):
        x = self.forward_features(x)
        return x
