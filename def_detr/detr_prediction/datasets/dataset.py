# Please do not change this file.
# We will use this file to benchmark your model.
# If you find a bug, post it on campuswire.

import os
import yaml
import random
from PIL import Image
import torch

class_dict = {
'plane': 0,
'ship': 1,
'storage-tank': 2,
'baseball-diamond': 3,
'tennis-court': 4,
'basketball-court': 5,
'ground-track-field': 6,
'harbor': 7,
'bridge': 8,
'large-vehicle': 9,
'small-vehicle': 10,
'helicopter': 11,
'roundabout': 12,
'soccer-ball-field': 13,
'swimming-pool': 14,
'container-crane': 15
}

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.transform = transform

        self.image_dir = root
        self.num_images = len(os.listdir(self.image_dir))
    
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of labeled image is from 0
        with open(os.path.join(self.image_dir, f"{idx}.tif"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transform(img)

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /labeled
            split: The split you want to used, it should be training or validation
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.transforms = transforms

        #self.image_dir = os.path.join(root, split, "0")
        #self.label_dir = os.path.join(root, split, "labels")

        self.image_dir = '../../Data/classifier_test/0'
        self.label_dir = '/home/ds6812/cvproj2/data/test/0_lbl'
        self.num_images = len(os.listdir(self.image_dir))
    
    def __len__(self):
        return self.num_images #self.num_images

    def __getitem__(self, idx):
        # the idx of training image is from 1 to 30000
        # the idx of validation image is from 30001 to 50000
    
        if self.split == "training":
            offset = 1
        if self.split == "validation":
            offset = 1

        #with open(os.path.join(self.image_dir, f"{idx + offset}.png"), 'rb') as f:
        #    img = Image.open(f).convert('RGB')
        img = Image.open(os.path.join(self.image_dir, f"{idx + offset}.tif")).convert('RGB')
        with open(os.path.join(self.label_dir, f"{idx + offset}.yml"), 'rb') as f:
            yamlfile = yaml.load(f, Loader=yaml.FullLoader)

        #img = img.resize((224,224 ))
        #print('SIZEEEEEEE///////////////////////')
        #print(img.size)
        w, h = img.size
        num_objs = len(yamlfile['labels'])
        #xmin, ymin, xmax, ymax
        boxes = torch.as_tensor(yamlfile['bboxes'], dtype=torch.float32)
        img_size = torch.as_tensor(yamlfile['image_size'], dtype=torch.float32)
        labels = []
        for label in yamlfile['labels']:
            labels.append(class_dict[label])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        #print('SORTINGGGGG DATAAA')
        #print(boxes.size)
        if(len(boxes.size()) == 1):
            if(boxes.numel() == 0):
                #print(f"{idx + offset}.png")
                area = torch.tensor([0])
            #print(boxes)
            #print(boxes.size())
            #print('--------------------------------')
            #boxes = boxes.unsqueeze(0)
            else:
                area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
                area = area.tolist()
                area = torch.tensor([area])
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #area = w * h
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])

        a = area.numpy()
        if(a[0] != 0):
            if self.transforms is not None:
                img, target = self.transforms(img, target)
        
        return img, target

