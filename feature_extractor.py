import os.path
import random
from typing import Tuple, List, Any

import numpy as np
import onnx
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models import (
    mnasnet0_75,
    MNASNet0_75_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    vgg16_bn,
    VGG16_BN_Weights,
)


class FeatureExtractor:
    """
    Class used to define a feature extractor by using pretrained model and visualize its outcomes.

    Attributes:
        pretrained_models (dict): dict with pretrained models' constructors and weights
        model (Any): pretrained model used for feature extraction
        image_transform (Any): image transformator associated with the pretrained model. Used to transform raw images.
        device (str): name of the device on which the calculations are performed
        last_layer_name (str): name of the last layer
    """

    pretrained_models = {
        "mnasnet0_75": {
            "model": mnasnet0_75,
            "weights": MNASNet0_75_Weights.IMAGENET1K_V1,
        },
        "mobilenet": {
            "model": mobilenet_v3_large,
            "weights": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        },
        "vgg": {
            "model": vgg16_bn,
            "weights": VGG16_BN_Weights.IMAGENET1K_V1,
        }
    }

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        """
        Initializes feature extractor's attributes

        Args:
            model_name (str): name of the pretrained model which will be used to extract features
            device (str): device (str): name of the device on which the calculations are performed
        """

        self.device: Any = torch.device(device)
        self.image_transform: Any = None
        self.last_layer_name: str = ""

        if model_name not in FeatureExtractor.pretrained_models:
            raise NotImplementedError(
                f"Given model name was not recognized. "
                f"It should be one of the following: {FeatureExtractor.pretrained_models.keys()}"
            )

        model_config = FeatureExtractor.pretrained_models[model_name]
        self.model = model_config["model"](weights=model_config["weights"])
        self.image_transform = model_config["weights"].transforms(antialias=True)
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False

        self.last_layer_name = self.list_all_layers()[-1]

    def list_all_layers(self, display: bool = False) -> List[str]:
        """
        A method used to list all the layers in the model

        Args:
            display (bool): should the nodes be printed to the console

        Returns:
            A list of strings with layers' names
        """
        nodes, _ = get_graph_node_names(self.model)

        if display:
            print(*[node for node in nodes], sep="\n")

        return nodes

    def slice_net(self, layer_name: str, overwrite_model: bool = False) -> Any:
        """
        A method extracting from the model the part of the model up until the layer with the given index (including)

        Args:
            layer_name (str): name of the layer after which the cutting is performed
            overwrite_model (bool): whether to overwrite the current model after slicing

        Returns:
            Feature extractor module after slicing
        """

        if layer_name not in self.list_all_layers():
            raise ValueError(f"Could not find layer of name {layer_name}. Be sure to use one of the names of the nodes")

        feature_extractor = create_feature_extractor(self.model, [layer_name])

        if overwrite_model:
            self.model = feature_extractor
            self.last_layer_name = layer_name

        return feature_extractor

    def get_image_from_file(self, path_to_image: str) -> Any:
        """
        Used for reading the image from the given path and transforming it using models' predefined transformation

        Args:
             path_to_image (str): path to the image to read

        Returns:
            Image after transformation
        """

        img = read_image(path_to_image)
        img = self.image_transform(img)  # C, H, W
        img = img.to(self.device)
        return img

    def feed(self, batch: Any) -> torch.Tensor:
        """
        Method used to get the outcome after feeding the pretrained model

        Args:
            batch (Any): the input with the expected shape (B, C, H, W) or  (C, H, W)

        Returns:
            A tensor as an output of the model. The shape is analogous to the input.
        """

        if self.model is None:
            raise AttributeError("Cannot feed model if model is None")
        return self.model(batch)[self.last_layer_name].to(dtype=torch.float)

    def save_feature_maps(self, path_to_image: str, path_to_folder: str, max_figs: int = 20) -> None:
        """
        Method used to save feature maps of the current net to the folder

        Args:
            path_to_image (str): path to the sample file used for visualization
            path_to_folder (str): path to the folder where the feature maps will be saved
            max_figs (int): a number of figures to save
        """

        try:
            img = self.get_image_from_file(path_to_image).unsqueeze(0)
            features = self.feed(img).squeeze().detach().numpy()  # Only first batch

            number_of_zeroes = int(np.ceil(len(features) ** .1))  # For the file name
            map_list = [i for i in range(min(features.shape[0], max_figs))]

            for idx in map_list:
                new_path = path_to_folder + "feature_map_layer" + f"{idx}".zfill(number_of_zeroes) + ".jpg"
                plt.imshow(features[idx], cmap="cividis")
                plt.axis("off")
                plt.savefig(new_path, bbox_inches="tight")

            plt.close("all")

        except Exception as e:
            print(f"Could not save features to the folder due to: {e}")

    def export_onnx(self, export_path: str, dummy_file_path: str) -> None:
        """
        A method used for exporting the model in the onnx format

        Args:
            export_path (str): path of the folder where the exported file will be saved
            dummy_file_path (str): path of the file used to determine the shapes and connections between layers
        """
        batch = self.get_image_from_file(dummy_file_path).unsqueeze(0)
        export_path = os.path.join(export_path, "feature_extractor.onnx")

        add_shape_info = False
        if not os.path.exists(export_path):
            torch.onnx.export(self.model, batch, export_path)
            add_shape_info = True

        if add_shape_info:
            onnx.save(onnx.shape_inference.infer_shapes(onnx.load(export_path)), export_path)

    @staticmethod
    def plot_feature_maps(features: torch.Tensor, plot_shape: Tuple = (5, 5), seed: int = None) -> None:
        """
        A method used for visualizing feature maps on the screen

        Args:
            features (torch.Tensor): output from the net
            plot_shape (tuple): shape of the plot
            seed (int): passed to the random generator. Used for reproducibility

        Raises:
            ValueError:
                When the given tensor has wrong dimensions,
                When the given shape consists of not positive values
        """

        try:
            features = features.detach().numpy()
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

    def plot_filters(self, layer_num: int, how_many: int, normalize: bool = True, seed: int = None) -> None:
        """
        A method used for plotting the filter shapes and weights of the last layer

        Args:
            layer_num (int): number of the layer which filters will be visualized
            how_many (int): number of filters to visualize
            normalize (bool): whether the weights should be normalized before visualization
            seed (int): passed to the random generator. Used for reproducibility

        Raises:
            ValueError: when the given number of filters to visualize is bigger than actual number of filters
        """

        try:
            filters = list(self.model.features.children())[layer_num].weight.detach().numpy()
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

                fig, axs = plt.subplots(how_many, how_many)
                im = None
                for row in range(how_many):
                    for column, ch_num in enumerate(random.sample(range(c_in), how_many)):
                        if row == 0:
                            axs[row, column].set_title(f"Channel {ch_num+1}")

                        im = axs[row, column].imshow(filters[row][ch_num], cmap="cividis")
                        axs[row, column].axis('off')
                colour_bar = plt.colorbar(im, ax=axs.ravel().tolist())
                colour_bar.outline.set_visible(False)
                plt.show()

        except Exception as e:
            print(f"Could not visualize filters due to: {e}")


if __name__ == '__main__':
    image_path = "imgs/surfing.jpg"

    fe = FeatureExtractor(model_name="vgg", device="cpu")
    fe.slice_net("features.32", overwrite_model=True)
    image = fe.get_image_from_file(image_path).unsqueeze(0)
    output = fe.feed(image)
    fe.plot_feature_maps(output, plot_shape=(5, 5))
    fe.plot_filters(3,  4)
