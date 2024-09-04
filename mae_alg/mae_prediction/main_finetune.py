import numpy as np
from addict import Dict
import os
import sys
import random
import math

import torch
import torch.backends.cudnn as cudnn

from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from Detector import models_vit
from Detector import vision_trans

import Detector.transforms as T
from eval import LabeledDataset, MetricLogger, SmoothedValue, evaluate

''' importing from Facebook repository'''
from Detector.util.pos_embed import interpolate_pos_embed
from Detector.util.lr_sched import warmup_lr_scheduler


def split_params(model_params, learning_rate=0.0001):
    '''
    Primarily works with blocks in ViT backbone
    Splits parameters, sets learning rates and weight decays and returns 
    a list to be appended to list of trainable parameters. 
    The list contains two dictionaries
    Dictionary 1: Contains learning rate and model paramters with keywords "norm" and "bias"
    Dictionary 2: Contains learning rate, weight_decay and all other model parameters

    Args:
        model_params:   The model parameters
        learning_rate:  The learning rate to be set while training for the model parameters
    Returns:
        List of dictionaries
    '''
    l1, l2 = list(), list()
    for n, p in model_params.named_parameters():
        if all([n.find(kw) == -1 for kw in ['norm', 'bias']]):
            l2 += [p]
        else:
            l1 += [p]

    with_keyword, without_keyword = {}, {}

    with_keyword['params'] = l1
    with_keyword['lr'] = learning_rate
    with_keyword['weight_decay'] = 0.0

    without_keyword['params'] = l2
    without_keyword['lr'] = learning_rate

    return [with_keyword, without_keyword]


def set_params_for_ViT_blocks(model, lr=0.0001, lr_ratio=0.69):
    '''
    The ViT model used as backbone is the vit_base_patch16, that has a depth of 12 (i.e. contains 12 blocks). 
    Sets the learning rate for each of the 12 blocks with the later blocks learning 
    progressively faster than the earlier ones

    The ViT model also has a patch_embed and pos_embed blocks which are set to have the smallest learning rate

    Args: 
        model:      The model to train containing ViT as the backbone
                    We assume that the model is also fitted with a Feature Pyramid Network that 
                    recives embeddings from ViT
        lr:         The base learning rate to set
        lr_ratio:   The learning rate to multiply with the base learning rate and exponentially decrease
                    Should be in range (0, 1)
    Returns:
        List of trainable parameters with their respective learning rates set
    '''
    params_to_train = list()
    params_to_train += split_params(
        model.backbone.backbone.blocks[11], learning_rate=lr * (lr_ratio))
    params_to_train += split_params(
        model.backbone.backbone.blocks[10], learning_rate=lr * (lr_ratio ** 2))
    params_to_train += split_params(
        model.backbone.backbone.blocks[9], learning_rate=lr * (lr_ratio ** 3))
    params_to_train += split_params(
        model.backbone.backbone.blocks[8], learning_rate=lr * (lr_ratio ** 4))
    params_to_train += split_params(
        model.backbone.backbone.blocks[7], learning_rate=lr * (lr_ratio ** 5))
    params_to_train += split_params(
        model.backbone.backbone.blocks[6], learning_rate=lr * (lr_ratio ** 6))
    params_to_train += split_params(
        model.backbone.backbone.blocks[5], learning_rate=lr * (lr_ratio ** 7))
    params_to_train += split_params(
        model.backbone.backbone.blocks[4], learning_rate=lr * (lr_ratio ** 8))
    params_to_train += split_params(
        model.backbone.backbone.blocks[3], learning_rate=lr * (lr_ratio ** 9))
    params_to_train += split_params(
        model.backbone.backbone.blocks[2], learning_rate=lr * (lr_ratio ** 10))
    params_to_train += split_params(
        model.backbone.backbone.blocks[1], learning_rate=lr * (lr_ratio ** 11))
    params_to_train += split_params(
        model.backbone.backbone.blocks[0], learning_rate=lr * (lr_ratio ** 12))
    params_to_train += split_params(model.backbone.backbone.patch_embed,
                                    learning_rate=lr * (lr_ratio ** 13))
    params_to_train.append({'params': model.backbone.backbone.pos_embed,
                           'lr': lr * (lr_ratio ** 13), 'weight_decay': 0.0})
    return params_to_train


def get_params(model, mode='none', lr=0.0001):
    '''
    Returns a list of model parameters that are to be updated during training
    Arguments: 
        model:  The loaded model
                We assume that the model is also fitted with a Feature Pyramid Network that 
                recives embeddings from ViT 
        mode:   Strategy to train the model
                    1. "none":      Default mode, all model parameters loaded and trained at same rate
                    2. "freeze":    All backbone parameters are frozen and not trained
                    3. "decay":     All model parameters are trained, but different groups trained at differnt rates
        lr:     Learning rate to use
    Returns: 
        List of model parameters that need to be trained
    '''
    if mode == 'none':
        return [p for p in model.parameters() if p.requires_grad]
    elif mode == 'freeze':
        for p in model.backbone.backbone.parameters():
            p.requires_grad = False
        return [p for p in model.parameters() if p.requires_grad]
    else:
        # Get the ViT backbone parameters
        backbone_ids = list(
            map(id, model.backbone.backbone.parameters()))

        # Get all model parameters that are not part of the ViT backbone
        other_params = filter(lambda p: id(
            p) not in backbone_ids, model.parameters())

        # Set learning rate for each parameter group
        # Group 1:
        param_groups = [{'params': other_params, 'lr': lr}]
        # Group 2: (ViT Norm parameters)
        param_groups.append(
            {'params': model.backbone.backbone.norm.parameters(), 'lr': lr, 'weight_decay': 0.0})
        # Group 3 (ViT blocks)
        param_groups += set_params_for_ViT_blocks(model, lr)

        return param_groups


def training_loop(model, optimizer, data_loader, device, epoch, print_freq=500, warmup_iter=8000):
    '''
    Runs a single epoch of training
    Adapted from Facebook Research code: https://github.com/facebookresearch/mae/blob/main/engine_pretrain.py
    Removed support for multi-GPU training

    Args: 
        model:          The model to train
        optimizer:      The optimizer with the learning rates set for each trainable parameter for the model
        data_loader:    The DataLoader object with the training data
        device:         The device that the training is to be run on (cuda preferred)
        epoch:          The epoch number of this training loop
        print_freq:     The print frequency to be fed to the metric_logger object 
        warmup_iter:    The number of batches for which learning_rate warmup should occur
    Returns:
        metric_logger:  Contains training loop information
    '''
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    # warmup learning rate if first epoch
    if epoch == 1:
        warmup_factor = 0.067
        # Ensure number of batchs for warmup is lesser than amount of training data available
        warmup_iters = min(
            warmup_iter, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # For each batch, move images and targets to device
        images = list(image.to(device) for image in images)
        targets = [
            {k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # No Multi-GPU support therefore loss_dict_reduced is equal to loss_dict
        loss_dict_reduced = loss_dict
        losses_reduced = sum(
            loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update metric_logger
        metric_logger.update(
            loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(
            lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def get_transform(train, use_jitter=False):
    transform_pipeline = []
    transform_pipeline.append(T.PILToTensor())
    if train:
        transform_pipeline.append(
            T.RandomHorizontalFlip(0.5))
        if use_jitter:
            transform_pipeline.append(T.Jitter())
    return T.Compose(transform_pipeline)


def get_model(path="checkpoints/ft_freeze-1.pth"):
    '''
    Note: Reuse of function defined in main_finetute.py
    The encoder in MAE is the VisionTransformer model that was trained in the upstream task
    This is loaded as the primary backbone
    The features extracted from the VisionTransformer model are passed to a feature pyramid network
    This model returns a dictionary of 5 sets of features of shape:
        {"0":   (Batch Size, 256, 128, 128)
         "1":   (Batch Size, 256, 64, 64)
         "2":   (Batch Size, 256, 32, 32)
         "3":   (Batch Size, 256, 16, 16)
         "pool":(Batch Size, 256, 8, 8)}
    This model is used as the backbone in the FasterRCNN's Region Proposal Network
    The weights saved after training are loaded into the model built and the model is returned

    Args: 
        path: The path to the saved checkpoint dict
    Returns:
        The model loaded from the checkpoint
    '''
    # Initializing the ViT model to expect images of size 512 X 512 and return embeddings of depth 768
    backbone = models_vit.__dict__['vit_base_patch16'](
        img_size=512, num_classes=100, drop_path_rate=0.1)

    # Add a Feature Pyramid network to work with different sizes of objects in the images
    backbone_with_fpn = vision_trans.VitDetAugmented(
        backbone=backbone)

    # Initializing the sizes and anchors FasterRCNN's Region Proposal Network
    # Sizes and aspect ratios should be of same length, thus each aspect_ratio list is multiplied by 5
    # Size list refers to the size w.r.t the original image
    size_list = [32, 64, 128, 256, 512]
    ratios = (0.5, 1.0, 2.0)
    ac_gen = AnchorGenerator(
        sizes=tuple((x,) for x in size_list),
        aspect_ratios=tuple([ratios] * 5))

    region_of_interest = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3", "pool"],
        output_size=8,
        sampling_ratio=2)

    # Initialize the FasterRCNN RPN backbone with the ViT model fitted with a Feature Pyramid Network
    model = FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=100,
        rpn_anchor_generator=ac_gen,
        box_roi_pool=region_of_interest
    )

    # Function to transform the input images to size 512 X 512
    model.transform = GeneralizedRCNNTransform(
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        min_size=512,
        max_size=512,
        fixed_size=(512, 512),
    )

    # Loading the weights saved after training into the model
    checkpoint = torch.load(path, map_location='cpu')
    print("loaded from ckpt: " + str(path))
    loaded_model = checkpoint['model']

    interpolate_pos_embed(
        model.backbone.backbone, loaded_model)
    msg = model.load_state_dict(loaded_model)

    return model


def main():
    '''
    Setup hyperparameters required for training.
    Loads the model and data and calls training loop for given epochs
    '''
    args = Dict({'exp_id': 'finetune',
                 'checkpoint_type': 'pretrained',
                 'checkpoint_path': './checkpoints/finetunedModel.pth',
                 'seed': 42,
                 'device': 'cuda',
                 'batch_size': 2,
                 'num_workers': 2,
                 'warmup_iter': 8000,
                 'use_jitter': False,
                 'lr': 0.0001,
                 'optim_mode': 'decay',
                 'scheduler_gamma': 0.1,
                 'scheduler_step_size': 6,
                 'num_epochs': 20,
                 'export_bound': 1,
                 'train_print_freq': 500})

    # Set seeds
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # Check if GPU is available and set device accordingly
    train_device = None
    if torch.cuda.is_available():
        train_device = torch.device('cuda')
    else:
        train_device = torch.device('cpu')

    # Load training and validation Datasets and add to DataLoader
    train_dataset = LabeledDataset(
        root='../../Data/labeled_data/',
        split='training',
        transforms=get_transform(
            train=True, use_jitter=args.use_jitter)
    )
    valid_dataset = LabeledDataset(
        root='../../Data/labeled_data/',
        split='validation',
        transforms=get_transform(train=False)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Initialize model and move to device
    model = get_model(args.checkpoint_path)
    model.to(train_device)
    print("model loaded to device")

    # init optimizer & scheduler
    optimizer = torch.optim.AdamW(get_params(
        model, mode=args.optim_mode), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    for epoch in range(1, args.num_epochs+1):
        # train for one epoch
        training_loop(model=model, optimizer=optimizer, data_loader=train_loader, device=train_device,
                      epoch=epoch, print_freq=args.train_print_freq, warmup_iter=args.warmup_iter)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the validation dataset
        evaluate(model, valid_loader, device=train_device)

        # save checkpoint every epoch
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
        ), 'epoch': epoch}, os.path.abspath('./checkpoints/{}-{}.pth'.format(args.exp_id, epoch)))


if __name__ == "__main__":
    main()
