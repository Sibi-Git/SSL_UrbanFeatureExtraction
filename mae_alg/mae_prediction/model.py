import torch

# imports for building the model before loading state_dict
from Detector import models_vit
from Detector import vision_trans
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from Detector.util.pos_embed import interpolate_pos_embed

def get_model(path = "./ft_resume_1-10.pth"):
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
    backbone = models_vit.__dict__['vit_base_patch16'](img_size=512, num_classes=16, drop_path_rate=0.1)

    # Add a Feature Pyramid network to work with different sizes of objects in the images
    backbone_with_fpn = vision_trans.VitDetAugmented(backbone=backbone)

    # Initializing the sizes and anchors FasterRCNN's Region Proposal Network
    # Sizes and aspect ratios should be of same length, thus each aspect_ratio list is multiplied by 5
    # Size list refers to the size w.r.t the original image
    size_list = [32, 64, 128, 256, 512]
    ratios = (0.5, 1.0, 2.0)
    ac_gen = AnchorGenerator(
        sizes=tuple((x,) for x in size_list),
        aspect_ratios=tuple([ratios] * 5))

    region_of_interest = MultiScaleRoIAlign(
        featmap_names = ["0", "1", "2", "3", "pool"],
        output_size = 8,
        sampling_ratio=2)

    # Initialize the FasterRCNN RPN backbone with the ViT model fitted with a Feature Pyramid Network
    model = FasterRCNN(
        backbone = backbone_with_fpn,
        num_classes = 16,
        rpn_anchor_generator = ac_gen,
        box_roi_pool = region_of_interest
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
    print("loaded from ckpt: "+ str(path))
    old_model = checkpoint['model']

    loaded_model = old_model.copy()
    for keys in old_model.keys():
        if(keys == 'rpn.head.conv.0.0.weight'):
            loaded_model['rpn.head.conv.weight'] = old_model['rpn.head.conv.0.0.weight']
            del loaded_model['rpn.head.conv.0.0.weight']
        if(keys == 'rpn.head.conv.0.0.bias'):
            loaded_model['rpn.head.conv.bias'] = old_model['rpn.head.conv.0.0.bias']
            del loaded_model['rpn.head.conv.0.0.bias']

    interpolate_pos_embed(model.backbone.backbone, loaded_model)
    msg = model.load_state_dict(loaded_model)

    return model
