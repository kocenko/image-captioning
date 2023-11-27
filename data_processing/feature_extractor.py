import os.path
from typing import List, Any

import numpy as np
import onnx
import matplotlib.pyplot as plt
import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from data_processing.pretrained_models import PRETRAINED_MODELS
from data_processing.image_transforms import ImageTransforms


class FeatureExtractor:
    """
    Class used to define a feature extractor by using pretrained model and visualize its outcomes.

    Attributes:
        model (Any): pretrained model used for feature extraction
        image_transform (Any): image transformator associated with the pretrained model. Used to transform raw images.
        last_layer_name (str): name of the last layer

    Args:
        model_name (str): name of the pretrained model which will be used to extract features
        image_transform (Any): used for transforming image before further processing
    """

    def __init__(self, model_name: str, image_transform: ImageTransforms) -> None:
        """Initializes feature extractor's attributes

        Args:
        """

        if model_name not in PRETRAINED_MODELS:
            raise NotImplementedError(
                f"Given model name was not recognized. "
                f"It should be one of the following: {PRETRAINED_MODELS.keys()}"
            )

        model_config = PRETRAINED_MODELS[model_name]
        self.model = model_config["model"](weights=model_config["weights"])
        self.image_transform = image_transform

        if torch.cuda.is_available():
            self.model.cuda()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.last_layer_name = self.list_all_layers()[-1]

    def list_all_layers(self) -> List[str]:
        """Lists all layers in the model

        Returns:
            A list of strings with layers' names
        """

        nodes, _ = get_graph_node_names(self.model)
        return nodes

    def slice_net(self, layer_name: str, overwrite_model: bool = False) -> Any:
        """Used for extracting from the model the part of the model up until the layer with the given name (including)

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

    @torch.no_grad()
    def feed(self, batch: Any) -> torch.Tensor:
        """Used to get the outcome of feeding the pretrained model

        Args:
            batch (Any): the input with the expected shape (B, C, H, W) or  (C, H, W)

        Returns:
            A tensor as an output of the model. The shape is analogous to the input.
        """

        if self.model is None:
            raise AttributeError("Cannot feed model if model is None")
        return self.model(batch)[self.last_layer_name].to(dtype=torch.float)

    def save_feature_maps(self, path_to_image: str, path_to_folder: str, max_figs: int = 20) -> None:
        """Used to save feature maps from the last layer of the current net to the folder

        Args:
            path_to_image (str): path to the sample file used for visualization
            path_to_folder (str): path to the folder where the feature maps will be saved
            max_figs (int): a number of figures to save
        """

        try:
            img = self.image_transform.read_image(path_to_image).unsqueeze(0)
            features = self.feed(img).squeeze().detach().numpy()  # Only first batch

            number_of_zeroes = int(np.ceil(len(features) ** 0.1))  # For the file name
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
        """Used for exporting the model in the onnx format

        Args:
            export_path (str): path of the folder where the exported file will be saved
            dummy_file_path (str): path of the file used to determine the shapes and connections between layers
        """
        batch = self.image_transform.read_image(dummy_file_path).unsqueeze(0)
        export_path = os.path.join(export_path, "feature_extractor.onnx")

        add_shape_info = False
        if not os.path.exists(export_path):
            torch.onnx.export(self.model, batch, export_path)
            add_shape_info = True

        if add_shape_info:
            onnx.save(onnx.shape_inference.infer_shapes(onnx.load(export_path)), export_path)


if __name__ == "__main__":
    pass
