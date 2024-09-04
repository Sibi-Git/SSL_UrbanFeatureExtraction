# 🌆 Extracting Urban Features Using Self-Supervised Learning (SSL)

## Table of Contents
1. [Overview](#1-overview)
2. [Explanation](#2-explanation)
3. [Getting Started](#3-getting-started)
    - [Swav Pretraining](#31-swav-pretraining)
    - [Deformable DETR Pretraining](#32-deformable-detr-pretraining)
    - [Deformable DETR Finetuning](#33-deformable-detr-finetuning)
    - [Deformable DETR Prediction](#34-deformable-detr-prediction)
    - [MAE Pretraining](#35-mae-pretraining)
    - [MAE Finetuning](#36-mae-finetuning)
    - [MAE Prediction](#37-mae-prediction)
    - [MAE Data Folders](#38-mae-data-folders)
4. [Data and Evaluation](#4-data-and-evaluation)
5. [Dataset Conversion](#5-dataset-conversion)
6. [Contributing](#6-contributing)
7. [Contact](#7-contact)

## 1. 🌍 Overview
This project explores the urban features influencing crime rates across various areas. By using satellite images, we aim to better understand urban environments and contribute insights to urban planning. The goal is to use object detection and classification techniques to extract features and correlate them with crime data.

We employ two advanced **self-supervised learning** (SSL) models:
- 🌀 **Deformable DETRreg**
- 🎭 **Masked Auto Encoder (MAE)**

The performance of these models is measured using:
- **Mean Average Precision (MAP)** for urban feature detection.
- **Classification Accuracy** of satellite image feature maps for high vs. low crime areas.

## 2. 🔍 Explanation

This project aims to study how urban features (such as road density, parks, and buildings) extracted from satellite images affect the crime rate in different areas. Understanding these relationships can help city planners and policymakers design safer urban environments.

We treat this problem as an **object detection** and **classification** task:
- **Object Detection**: Identifying urban features like roads, parks, etc., from satellite images using object detection models.
- **Classification**: Once features are detected, we classify areas as either high or low crime based on those features.

**Self-supervised learning (SSL)** is ideal for this task due to the vast availability of unlabeled satellite images. The models we use don’t require manual labeling, making them efficient for large-scale satellite data.

### Models Used
1. **Deformable DETRreg**: A deformable version of the DETR (DEtection TRansformers) model for object detection.
2. **Masked Auto Encoder (MAE)**: This model learns to reconstruct the missing parts of images (masked input), providing a self-supervised learning framework.

Both models are compared based on their ability to detect urban features (measured by Mean Average Precision) and how well these features classify areas into high or low crime categories.

---

## 3. 🚀 Getting Started

### 3.1 **Swav Pretraining**

Navigate to the `swav_pretrain` folder to initiate the pretraining process:

```bash
cd def_detr/swav_pretrain
git clone "https://github.com/NVIDIA/apex"
cd apex
git checkout 4a1aa97e31ca87514e17c3cd3bbc03f4204579d0
python3 setup.py install --cuda_ext
```
### ✅ Verify setup:

```bash
python3 -c 'import apex; from apex.parallel import LARC'
python3 -c 'import apex; from apex.parallel import SyncBatchNorm; print(SyncBatchNorm.__module__)'
```

### 🚀 Start Swav pretraining:

```bash
cd ..
python3 -m torch.distributed.launch --nproc_per_node=1 main_swav.py\
--data_path /unlabeled\
--epochs 15\
--base_lr 0.6\
--final_lr 0.006\
--warmup_epochs 0\
--batch_size 64\
--size_crops 224 96\
--nmb_crops 2 6\
--min_scale_crops 0.14 0.05\
--max_scale_crops 1.0 0.14\
--use_fp16 true\
--freeze_prototypes_niters 5005\
--queue_length 3840\
--checkpoint_freq 1\
--workers 2`
```

### 3.2 🌀 Deformable DETR Pretraining

To pretrain the **Deformable DETRreg**, navigate to the pretraining folder and run:

```bash
cd def_detr/detr_pretrain/models/ops
sh ./make.sh
cd ../..
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_top30_coco.sh --batch_size 8 --epochs 10 --num_workers 2`
```

### 3.3 🌀 Deformable DETR Finetuning

To fine-tune the Deformable DETR model:

```bash
cd def_detr/detr_finetune/models/ops
sh ./make.sh
cd ../..
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 ./configs/DETReg_fine_tune_full_coco.sh --batch_size 2 --epochs 50 --num_workers 2`
```

### 3.4 🌀 Deformable DETR Prediction

To run predictions using the Deformable DETR model:

```bash
cd def_detr/detr_prediction
python3 evaluate.py
```

### 3.5 🎭 MAE Pretraining

For pretraining the **Masked Auto Encoder (MAE)**:


```bash
cd mae/mae_pretraining
python3 main_pretrain.py
```

### 3.6 🎭 MAE Finetuning

Fine-tune the MAE model by running:

```bash
cd mae/mae_finetuning
python3 main_finetune.py ft_freeze
```

### 3.7 🎭 MAE Prediction

To predict using the MAE model:

```bash
cd mae/mae_prediction
python3 imageCheck.py
```

### 3.8 🎭 MAE Data Folders

The `mae_data/` folder contains the following subfolders:

-   **mae_finetuning**: Contains Python scripts and compiled Python files used for finetuning the MAE model.
-   **mae_pretraining**: Contains the scripts and data for pretraining the MAE model.
-   **mae_prediction**: Holds Python files necessary for running predictions using the MAE model.

* * * * *

📊 4. Data and Evaluation
-------------------------

The `Data/` folder contains the datasets used for training and evaluation. Results and logs can be found in the `Evaluation/` folder.

💻 5. Dataset Conversion
------------------------

In the `dataset_conversion/` folder, you can find scripts to convert raw datasets into the format required for this project.

🤝 6. Contributing
------------------

Feel free to raise issues or submit pull requests to improve this project!

📧 7. Contact
-------------

For any inquiries, you can contact me at:

-   **Sibi MARAPPAN**
-   📧 Email: msibi.mail@gmail.com
-   🔗 [LinkedIn Profile](https://www.linkedin.com/in/sibi-marappan/)
