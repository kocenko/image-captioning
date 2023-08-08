import os.path
import random
from typing import Tuple, List, Dict

import numpy as np
from sklearn import preprocessing
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.io import read_image
from torchvision.models import mnasnet0_75, MNASNet0_75_Weights
import onnx


def unravel_net(
        group: nn.Sequential,
        path: List[int] = None,
        mapping: Dict[int, List[int]] = None,
        available_index: int = 0
) -> Tuple[Dict[int, List[int]], int]:

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


def get_layer(net: nn.Sequential, which: int):
    mapping, last_idx = unravel_net(net)

    if which < 0 or which >= last_idx:
        raise ValueError(f"{which=}".split('=')[0] + f" parameter should be between {0} and {last_idx}")
    layer = net

    for index in mapping[which]:
        layer = list(layer.children())[index]

    return layer


def slice_net(net: nn.Sequential, after_which: int) -> nn.Sequential:
    mapping, last_idx = unravel_net(net)

    if after_which < 0 or after_which > last_idx:
        raise ValueError(f"{after_which=}".split('=')[0] + f" parameter should be between {0} and {last_idx}")
    subnet = [net]

    indices_list = mapping[after_which]
    for index in indices_list:
        subnet = subnet[:-1] + list(subnet[-1].children())[:index+1]

    return nn.Sequential(*subnet)


def save_feature_maps(features: np.ndarray, path_to_folder: str, order_by_brightness: bool = True):
    # Expecting features to be in shape (C, H, W)

    brightness_list = []
    for id, single_map in enumerate(features):
        brightness_list.append((id, np.mean(single_map)))

    if order_by_brightness:
        brightness_list.sort(key=lambda x: x[1], reverse=True)

    file_name = "feature_map_layer_2_sorted_"
    number_of_zeroes = int(np.ceil(len(features)**.1))
    for num, (id, _) in enumerate(brightness_list):
        new_path = path_to_folder + file_name + f"{num}".zfill(number_of_zeroes) + ".jpg"
        image = Image.fromarray((preprocessing.normalize(features[id]) * 255).astype(np.uint8), mode='L')
        image.save(new_path)


if __name__ == '__main__':
    # Initialization
    img = read_image("imgs/rooster.jpg")
    weights = MNASNet0_75_Weights.DEFAULT
    trained_model = mnasnet0_75(weights=weights)
    trained_model.train(False)
    model = trained_model.layers

    # Preprocessing the image
    transformator = weights.transforms(antialias=True)
    img = transformator(img)
    batch = img.unsqueeze(0)  # (B, C, H, W)

    # Slicing model and extracting last layer
    which_layer = 24
    last_layer = get_layer(model, which_layer)
    model = slice_net(model, which_layer)

    # Exporting model to visualize
    model_export_name = 'mnasnet0_75_sliced.onnx'
    add_shape_info = False
    if not os.path.exists(model_export_name):
        torch.onnx.export(model, batch, model_export_name)
        add_shape_info = True

    if add_shape_info:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model_export_name)), model_export_name)

    # Visualizing filters (weights)
    try:
        filters = last_layer.weight
        if filters.shape[2:] == torch.Size([1, 1]):
            print("Filters' kernel size is 1x1, so no need in visualizing")
        else:
            print("Visualizing filters not implemented yet")
    except AttributeError:
        print("Given layer does not have weights")

    # Visualizing feature map and saving to folder
    folder_path = "./feature_maps/"
    outcome = model(batch).detach().numpy()
    # save_feature_maps(outcome[0], folder_path)
    squares_along_axis = 5
    try:
        maps_ids = random.sample(range(outcome.shape[1]), squares_along_axis**2)
    except ValueError:
        print("Could not create a plot with given squares number due to the wrong shape of the outcome")
        squares_along_axis = int(outcome.shape[1]**.5)
        maps_ids = random.sample(range(outcome.shape[1]), squares_along_axis**2)

    maps = outcome[0, maps_ids, :, :]

    fig, axs = plt.subplots(squares_along_axis, squares_along_axis)
    for row in range(squares_along_axis):
        for column in range(squares_along_axis):
            axs[row, column].imshow(maps[row * squares_along_axis + column])
            axs[row, column].axis('off')
    plt.show()
