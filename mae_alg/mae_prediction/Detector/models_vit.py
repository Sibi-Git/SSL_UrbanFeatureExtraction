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

import torch
import torch.nn as nn

import timm.models.vision_transformer

'''
Obtained from https://github.com/facebookresearch/mae/blob/main/models_vit.py & 
https://github.com/ViTAE-Transformer/ViTDet/blame/main/mmdet/models/backbones/vit.py
'''


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ 
    Vision Transformer with support for global average pooling
    """

    def __init__(self, embed_dim=768, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.out_channels = embed_dim
        self.head = nn.Identity()

    def forward_features(self, x):
        # Recive input of shape B, 3, 512, 512 and convert to shape 3, 1024, 768
        x = self.patch_embed(x)
        B, dim1, _ = x.shape

        # Change input shape to B, dim1 + 1, 768 incorporating cls_tokens
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # Change the input shape back to B, dim1, 768 by removing cls_tokens
        x = self.norm(x)[:, 1:, :]  # remove cls token

        # Change shape of input to 4 dimesion for passing through ConvTranspose2d
        # B, dim1, 768 -> B, 768, dim1 -> B, 768, sqrt(dim1), sqrt(dim1)
        x = x.permute(0, 2, 1).reshape(
            B, -1, int(dim1 ** 0.5), int(dim1 ** 0.5))

        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
