from torchvision.models import (
    mnasnet0_75,
    MNASNet0_75_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    vgg16_bn,
    VGG16_BN_Weights,
)
from transformers import ViTImageProcessor, ViTModel


PRETRAINED_MODELS = {
    "mnasnet0_75": {
        "model": mnasnet0_75(weights=MNASNet0_75_Weights.IMAGENET1K_V1),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "transforms": MNASNet0_75_Weights.IMAGENET1K_V1.transforms(
                antialias=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    },
    "mobilenet": {
        "model": mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "transforms": MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms(
                antialias=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    },
    "vgg": {
        "model": vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "transforms": VGG16_BN_Weights.IMAGENET1K_V1.transforms(
                antialias=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
    },
    "vit": {
        "model": ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', add_pooling_layer=False),
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "transforms": ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'),
    },
}
