import os.path
import random
from typing import Tuple, List, Dict, Any, Union

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.io import read_image
from torchvision.models import MNASNet, mnasnet0_75, MNASNet0_75_Weights
import onnx


class FeatureExtractor:
    def __init__(self, model_name: str = "mnasnet0_75"):
        self.model = None
        self.mapping: Dict[int, List[int]] = {}
        self.available_layer_index: int = 0
        self.image_transform = None

        if model_name == "mnasnet0_75":
            weights = MNASNet0_75_Weights.DEFAULT
            self.model: MNASNet = mnasnet0_75(weights=weights)
            self.model.train(False)
            self.model = self.model.layers
            self.image_transform: Any = weights.transforms(antialias=True)
            self.generate_mapping()
        else:
            raise NotImplementedError

    def unravel_net(self, group: nn.Sequential, path: List[int] = None) -> None:

        if path is None:
            path = []

        for layer_number, layer in enumerate(list(group.children())):
            current_path = path + [layer_number]
            self.mapping[self.available_layer_index] = current_path

            # If Sequential or InverseResidual
            if len(list(layer.children())) > 0:
                self.unravel_net(layer, current_path)
            else:
                self.available_layer_index = self.available_layer_index + 1

    def generate_mapping(self, force: bool = False) -> None:
        if not bool(self.mapping) or (bool(self.mapping) and force):
            self.available_layer_index = 0
            self.unravel_net(self.model)
        else:
            print("Mapping was already performed. Change 'force' parameter to True.")

    def get_layer(self, layer_index: int) -> Union[nn.Sequential, nn.Module]:
        if not bool(self.mapping):
            raise AttributeError("To get layer you need to generate mapping first using 'generate mapping' method.")

        if layer_index < 0 or layer_index >= self.available_layer_index:
            raise ValueError(f"Method takes an argument with value between {0} and {self.available_layer_index}")

        layer = self.model
        for index in self.mapping[layer_index]:
            layer = list(layer.children())[index]

        return layer

    def slice_net(self, layer_index: int):
        if not bool(self.mapping):
            raise AttributeError("To slice net you need to generate mapping first using 'generate mapping' method.")

        if layer_index < 0 or layer_index >= self.available_layer_index:
            raise ValueError(f"Method takes an argument with value between {0} and {self.available_layer_index}")

        subnet = [self.model]
        indices_list = self.mapping[layer_index]
        for index in indices_list:
            subnet = subnet[:-1] + list(subnet[-1].children())[:index + 1]

        self.model = nn.Sequential(*subnet)
        self.generate_mapping(True)

    def get_image_from_file(self, path_to_image: str):
        img = read_image(path_to_image)
        img = self.image_transform(img)
        batch = img.unsqueeze(0)  # (B, C, H, W)
        return batch

    def feed(self, batch):
        if self.model is None:
            raise AttributeError("Cannot feed model if model is None")
        return self.model(batch).detach().numpy()

    def save_feature_maps(self, path_to_image: str, path_to_folder: str, order_by_mean: bool = True):
        try:
            img = self.get_image_from_file(path_to_image)
            features = self.feed(img)[0]  # Only first batch

            file_name = f"feature_map_layer_{self.available_layer_index-1}"
            number_of_zeroes = int(np.ceil(len(features) ** .1))  # For the file name
            map_list = [i for i in range(features.shape[0])]

            if order_by_mean:
                mean_list = [(map_list[index], np.mean(single_map)) for index, single_map in enumerate(features)]
                mean_list.sort(key=lambda x: x[1], reverse=True)
                map_list = [i[0] for i in mean_list]
                # print([i[1] for i in mean_list])
                file_name += "_sorted_"

            for idx in map_list:
                new_path = path_to_folder + file_name + f"{idx}".zfill(number_of_zeroes) + ".jpg"
                image = Image.fromarray((features[idx] * 255).astype(np.uint8))
                image.save(new_path)

        except Exception as e:
            print(f"Could not save features to the folder due to: {e}")

    def export_onnx(self, export_path: str, dummy_file_path: str):
        batch = self.get_image_from_file(dummy_file_path)

        export_path = export_path + f"_sliced_at_{self.available_layer_index-1}.onnx"

        add_shape_info = False
        if not os.path.exists(export_path):
            torch.onnx.export(self.model, batch, export_path)
            add_shape_info = True

        if add_shape_info:
            onnx.save(onnx.shape_inference.infer_shapes(onnx.load(export_path)), export_path)

    @staticmethod
    def plot_feature_maps(features: torch.Tensor, plot_shape: Tuple = (5, 5), seed: int = None):
        try:
            number_of_maps = features.shape[1]
            if number_of_maps == 0:
                raise ValueError("Tensor's 2nd dimension has 0 maps")

            if plot_shape[0] <= 0 or plot_shape[1] <= 0:
                raise ValueError("Plot shape should consist of positive values")

            plot_shape = list(plot_shape)
            changed_shape = False
            while number_of_maps < plot_shape[0] * plot_shape[1]:
                changed_shape = True
                if plot_shape[0] == 1:
                    plot_shape[1] -= 1
                else:
                    plot_shape[0] -= 1

            if changed_shape:
                print(f"Changed shape of the plot to ({plot_shape[0]}, {plot_shape[1]}) to fit the data")

            random.seed(seed)
            maps_ids = random.sample(range(number_of_maps), plot_shape[0] * plot_shape[1])

            maps = features[0, maps_ids, :, :]
            fig, axs = plt.subplots(plot_shape[0], plot_shape[1])
            im = None
            for row in range(plot_shape[0]):
                for column in range(plot_shape[1]):
                    im = axs[row, column].imshow(maps[row * plot_shape[0] + column], cmap="cividis")
                    axs[row, column].axis('off')
            colour_bar = plt.colorbar(im, ax=axs.ravel().tolist())
            colour_bar.outline.set_visible(False)
            plt.show()

        except Exception as e:
            print(f"Could not plot features due to: {e}")

    def plot_filters(self, layer_idx: int, how_many: int, normalize: bool = True, seed: int = None):
        try:
            filters = self.get_layer(layer_idx).weight.detach().numpy()
            c_out, c_in, h, w = filters.shape  # (Channels_out, Channels_in/Groups, K_height, K_width)

            if how_many > c_out:
                raise ValueError("There are not so many filters")

            if h == 1 and w == 1:
                print("Filters' kernel size is 1x1, so no need in visualizing")
            else:
                if normalize:
                    # MINMAX normalization
                    filter_min, filter_max = filters.min(), filters.max()
                    filters = (filters - filter_min) / (filter_max - filter_min)

                random.seed(seed)
                filters_ids = random.sample(range(c_out), how_many)
                filters = filters[filters_ids, :, :, :]

                fig, axs = plt.subplots(how_many, c_in)
                im = None
                for row in range(how_many):
                    for column in range(c_in):
                        if row == 0:
                            axs[row, column].set_title(f"Channel {column+1}")

                        im = axs[row, column].imshow(filters[row][column], cmap="cividis")
                        axs[row, column].axis('off')
                colour_bar = plt.colorbar(im, ax=axs.ravel().tolist())
                colour_bar.outline.set_visible(False)
                plt.show()

        except Exception as e:
            print(f"Could not visualize filters due to: {e}")


if __name__ == '__main__':
    image_path = "./exercise/imgs/rooster.jpg"
    model_export_name = './onnx_models/mnasnet0_75'
    folder_path = "./feature_maps/"

    fe = FeatureExtractor()
    fe.slice_net(50)
    image = fe.get_image_from_file(image_path)
    output = fe.feed(image)
    # fe.export_onnx(model_export_name, image_path)
    fe.plot_feature_maps(output, plot_shape=(2, 2))
    fe.plot_filters(20, 4)
