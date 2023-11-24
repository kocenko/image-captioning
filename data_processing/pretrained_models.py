from torchvision.models import (
    mnasnet0_75,
    MNASNet0_75_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    vgg16_bn,
    VGG16_BN_Weights,
)

PRETRAINED_MODELS = {
    "mnasnet0_75": {
        "model": mnasnet0_75,
        "weights": MNASNet0_75_Weights.IMAGENET1K_V1,
    },
    "mobilenet": {
        "model": mobilenet_v3_small,
        "weights": MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    },
    "vgg": {
        "model": vgg16_bn,
        "weights": VGG16_BN_Weights.IMAGENET1K_V1,
    },
}
