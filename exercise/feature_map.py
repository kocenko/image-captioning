import os.path
from typing import Tuple, Union, List

import torch
from torch import nn
from torchvision.io import read_image
from torchvision.models import mnasnet, mnasnet0_75, MNASNet0_75_Weights
import onnx


def unravel_net(group: nn.Sequential, path: List = None, mapping: dict = None, available_index: int = 0) -> Tuple[dict, int]:
    if path is None:
        path = []

    if mapping is None:
        mapping = {}

    layer_idx = available_index
    for layer_number, layer in enumerate(list(group.children())):
        current_path = path + [layer_number]
        mapping[layer_idx] = current_path

        # If Sequential or InverseResidual
        if len(list(layer.children())) > 0:
            mapping, layer_idx = unravel_net(layer, current_path, mapping.copy(), layer_idx)
        else:
            layer_idx += 1

    return mapping, layer_idx


def slice_net(net: nn.Sequential, after_which: int) -> nn.Sequential:
    mapping, last_idx = unravel_net(net)

    if after_which < 1 or after_which > last_idx:
        raise ValueError(f"{after_which=}".split('=')[0] + f" parameter should be more than {0} and less than {last_idx+1}")
    subnet = [net]

    indices_list = mapping[after_which-1]
    for index in indices_list:
        subnet = subnet[:-1] + list(subnet[-1].children())[:index+1]

    return nn.Sequential(*subnet)


def map_net(net: nn.Sequential):
    new_net = slice_net(net, 5)
    print(new_net)


if __name__ == '__main__':
    # Initialization
    img = read_image("imgs/surfing.jpg")
    weights = MNASNet0_75_Weights.DEFAULT
    trained_model = mnasnet0_75(weights=weights)
    model = trained_model.layers
    map_net(model)

    # Preprocessing the image
    transformator = weights.transforms(antialias=True)
    image = transformator(img)
    batch = image.unsqueeze(0)  # (B, C, H, W)

    # Exporting model to visualize
    model_export_name = 'mnasnet0_75.onnx'
    add_shape_info = False
    if not os.path.exists(model_export_name):
        torch.onnx.export(model, batch, model_export_name)
        add_shape_info = True

    if add_shape_info:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_export_name)), model_export_name)
