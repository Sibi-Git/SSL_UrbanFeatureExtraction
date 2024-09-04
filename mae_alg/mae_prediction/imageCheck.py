from eval import *
from main_finetune import *
from model import *
from PIL import Image
import PIL
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2

model = get_model()
model.eval()

final_labels = []

for file_name in os.listdir('../../Data/classifier_test/0/'):

	lbl = []

	im = Image.open("../../Data/classifier_test/0/" + file_name).convert('RGB')
	transform = T.Compose([
		T.Resize((224, 224)),
		T.ToTensor(),
		T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		])

	X = transform(im)

	print(file_name)

	conv_features, enc_attn_weights, dec_attn_weights = [], [], []

	hooks = [
		model.backbone.register_forward_hook(
			lambda self, input, output: conv_features.append(output)
		),
	]

	predictions = model(X.unsqueeze(0))

	for hook in hooks:
		hook.remove()
			
	conv_features = conv_features[0]
	#enc_attn_weights = enc_attn_weights[0]
	#dec_attn_weights = dec_attn_weights[0]
	#h, w = conv_features['0'].tensors.shape[-2:]

	#print(h)
	#print(w)

	#print(list(conv_features.items())[0][1].cpu().detach().numpy().reshape(128,128).shape)

	cv2.imwrite('./classifier_train_featuremaps/featuremaps/0fm/' + file_name, list(conv_features.items())[0][1].cpu().detach().numpy().reshape(2048,2048))

	for i in range(len(predictions)):
		lbl.append(predictions[i]['labels'])

	final_labels.append(lbl)

	#break

with open("./classifier_train_featuremaps/labels/0lbl.txt", "w") as output:
	output.write(str(final_labels))
