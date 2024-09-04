import torch
from datasets.coco_utils import get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from datasets.dataset import LabeledDataset
from argparse import Namespace
from models import build_model
import util.misc as utils
from engine import evaluate

import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

BATCH_SIZE = 2
NUM_WORKERS = 2

ARGS = {
    'lr': 0.0002,
    'max_prop': 30,
    'lr_backbone_names': ['backbone.0'],
    'lr_backbone': 2e-05,
    'lr_linear_proj_names': ['reference_points', 'sampling_offsets'],
    'lr_linear_proj_mult': 0.1,
    'batch_size': 4,
    'weight_decay': 0.0001,
    'epochs': 50,
    'lr_drop': 40,
    'lr_drop_epochs': None,
    'clip_max_norm': 0.1,
    'sgd': True,
    'filter_pct': -1,
    'with_box_refine': False,
    'two_stage': False,
    'strategy': 'topk',
    'obj_embedding_head': 'intermediate',
    'frozen_weights': None,
    'backbone': 'resnet50',
    'dilation': True,
    'position_embedding': 'sine',
    'position_embedding_scale': 6.283185307179586,
    'num_feature_levels': 4,
    'enc_layers': 6,
    'dec_layers': 6,
    'dim_feedforward': 1024,
    'hidden_dim': 256,
    'dropout': 0.1,
    'nheads': 8,
    'num_queries': 300,
    'dec_n_points': 4,
    'enc_n_points': 4,
    'pretrain': '',
    'load_backbone': 'swav',
    'masks': False,
    'aux_loss': True,
    'set_cost_class': 2,
    'set_cost_bbox': 5,
    'set_cost_giou': 2,
    'object_embedding_coeff': 1,
    'mask_loss_coef': 1,
    'dice_loss_coef': 1,
    'cls_loss_coef': 2,
    'bbox_loss_coef': 5,
    'giou_loss_coef': 2,
    'focal_alpha': 0.25,
    'dataset_file': 'coco',
    'dataset': 'coco',
    'data_root': 'data',
    'coco_panoptic_path': None,
    'remove_difficult': True,
    'output_dir': 'out',
    'cache_path': 'cache/ilsvrc/ss_box_cache',
    'device': 'cuda',
    'seed': 42,
    'resume': '',
    'eval_every': 1,
    'start_epoch': 0,
    'eval': True,
    'viz': True,
    'num_workers': 2,
    'cache_mode': False,
    'object_embedding_loss': False
}


if __name__ == "__main__":
    args = Namespace(**ARGS)

    print("Starting evaluation")
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit()
    device = torch.device('cuda')

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=make_coco_transforms("test"))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=utils.collate_fn)

    model, _, postprocessors = build_model(args)
    model.to(device)
    model.load_state_dict(torch.load("./model_weights.pth")["model"], strict=True)

    evaluate(model, postprocessors, valid_loader, device)