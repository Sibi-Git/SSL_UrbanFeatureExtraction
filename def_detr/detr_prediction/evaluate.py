import torch
import time
import os
from datasets.coco_utils import get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from datasets.dataset import LabeledDataset
from argparse import Namespace
from models import build_model
import util.misc as utils
from engine import evaluate
from PIL import Image
from datasets.coco_eval import CocoEvaluator
import torchvision.transforms as T
import matplotlib.pyplot as plt
import sys
import cv2
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
	final = []
	args = Namespace(**ARGS)

	print("Starting evaluation")
	if not torch.cuda.is_available():
		print("CUDA required")
		sys.exit()
	device = torch.device('cuda')

	model, _, postprocessors = build_model(args)
	model.to(device)
	model.load_state_dict(torch.load("./model_weights.pth")["model"], strict=True)

	model.eval()

	def box_cxcywh_to_xyxy(x):
		x_c, y_c, w, h = x.unbind(1)
		b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
		return torch.stack(b, dim=1)

	def rescale_bboxes(out_bbox, size):
		img_w, img_h = size
		b = box_cxcywh_to_xyxy(out_bbox)
		b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
		return b

	for file_name in os.listdir('../../Data/classifier_test/1/'):

		im = Image.open("../../Data/classifier_test/1/" + file_name).convert('RGB')


		print(file_name)
		transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
		# for output bounding box post-processing

		# mean-std normalize the input image (batch-size: 1)
		img = transform(im).unsqueeze(0)

		conv_features, enc_attn_weights, dec_attn_weights = [], [], []

		#print('backboneeee')

		#print(list(model.backbone))

		hooks = [
				model.backbone[-2].register_forward_hook(
					lambda self, input, output: conv_features.append(output)
				),
				model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
					lambda self, input, output: enc_attn_weights.append(output)
				),
				model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
					lambda self, input, output: dec_attn_weights.append(output)
				),
		]

		# propagate through the model
		outputs = model(img.to(device))

		for hook in hooks:
				hook.remove()
				
		conv_features = conv_features[0]
		enc_attn_weights = enc_attn_weights[0]
		dec_attn_weights = dec_attn_weights[0]

		# keep only predictions with 0.7+ confidence
		probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
		keep = probas.max(-1).values > 0.0

		# convert boxes from [0; 1] to image scales
		bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].to(device), im.size)

		h, w = conv_features['0'].tensors.shape[-2:]

		for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), bboxes_scaled):
			#cv2.imwrite('/home/ds6812/cvproj2/detr_features/1.jpg', dec_attn_weights[0, idx].view(h, w))
			cv2.imwrite('./classifier_train_featuremaps/training/1fm/' + file_name, dec_attn_weights[0, idx].cpu().detach().numpy().reshape(16, 16))

		new_image = Image.open('./classifier_train_featuremaps/training/1fm/' + file_name)
		new_image = new_image.resize((100, 100))
		new_image.save('./classifier_train_featuremaps/training/1fm/' + file_name)

		lbls = []
		for p, (xmin, ymin, xmax, ymax) in zip(probas[keep], bboxes_scaled.tolist()):
					lbls.append(torch.argmax(p).item())

		final.append(lbls)

with open("1lbl.txt", "w") as output:
		output.write(str(final))
