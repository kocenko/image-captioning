from typing import Callable
from collections import defaultdict
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import torch
import networkx as nx

from model.transformer import CandidateGraph
from evaluation.beam_search_util import unravel_graph


def plot_numpy_logs(name: str, folder: str = "../numpy_logs", batch_num: int = 0) -> None:
    """Used to display 3d and 4d tensors saved as numpy arrays

    Args:
        name (str): name of the array (between the first '_' and extension in the file's name)
        folder (str): path to the folder where numpy arrays are saved
        batch_num (int): index of the batch to visualize (based on the first dimension)
    """

    all_files = os.listdir(folder)
    all_numpy = [file.split(".", maxsplit=1)[0] for file in all_files if file.split(".", maxsplit=1)[1] == "npy"]
    all_match = [os.path.join(folder, f"{file}.npy") for file in all_numpy if file.split("_", maxsplit=1)[1] == name]
    assert len(all_match) == 2, "Expected two files with the given name"

    array1 = np.load(all_match[0])
    array2 = np.load(all_match[1])

    assert (
        array1.shape == array2.shape
    ), f"Shapes {array1.shape} of {all_match[0]} and {array2.shape} of {all_match[1]} do not match"
    assert batch_num < array1.shape[0], "Wrong batches num"

    n_cols = 2
    if len(array1.shape) == 4:
        n_rows = array1.shape[1]
        array1 = [array1[batch_num][i] for i in range(n_rows)]
        array2 = [array2[batch_num][i] for i in range(n_rows)]
    elif len(array1.shape) == 3:
        n_rows = 1
        array1 = [array1[batch_num]]
        array2 = [array2[batch_num]]
    else:
        raise Exception("Wrong shapes")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            if j == 0:
                ax.set_ylabel(f"{i}")
                ax.imshow(array1[i])
            if j == 1:
                ax.imshow(array2[i])
            if i == 0:
                ax.set_title(f"{all_match[j]}")

    plt.tight_layout()
    plt.show()


def plot_feature_maps(features: torch.Tensor, plot_shape: tuple[int, int] = (5, 5), seed: int = None) -> None:
    """Used for visualizing feature maps

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
                axs[row, column].axis("off")
        colour_bar = plt.colorbar(im, ax=axs.ravel().tolist())
        colour_bar.outline.set_visible(False)
        plt.show()

    except Exception as e:
        print(f"Could not plot features due to: {e}")


def plot_filters(filters: np.ndarray, how_many: int, normalize: bool = True, seed: int = None) -> None:
    """Used for plotting the filter shapes and weights of the last layer

    Args:
        filters (np.ndarray): filters to visualize
        how_many (int): number of filters to visualize
        normalize (bool): whether the weights should be normalized before visualization
        seed (int): passed to the random generator. Used for reproducibility

    Raises:
        ValueError: when the given number of filters to visualize is bigger than actual number of filters
    """

    try:
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
                    axs[row, column].axis("off")
            colour_bar = plt.colorbar(im, ax=axs.ravel().tolist())
            colour_bar.outline.set_visible(False)
            plt.show()

    except Exception as e:
        print(f"Could not visualize filters due to: {e}")


def plot_captioned_image(image: torch.Tensor, generated_caption: str, show: bool = False):
    fig, ax = plt.subplots(1)
    plt.axis("off")
    ax.imshow(image)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
    ax.text(
        image.shape[1] // 2,
        image.shape[0] + 0.02,
        f"{generated_caption}",
        ha="center",
        va="center",
        size=10,
        bbox=bbox_props,
    )
    if show:
        plt.show()
    return fig


def plot_self_attention(
    base_image: torch.Tensor,
    attention_weights: list[list[torch.Tensor]],
    layers_num: int,
    heads_num: int,
    patches_per_axis: int,
    show: bool = False,
):
    if layers_num > len(attention_weights):
        print(f"Cannot visualize more layers than {len(attention_weights)}")
        layers_num = len(attention_weights)
    layer_step = len(attention_weights) // layers_num
    layers_ids = list(range(0, len(attention_weights), layer_step))
    rows_num = layers_num

    if heads_num > len(attention_weights[0]):
        print(f"Cannot visualize more heads than {len(attention_weights[0])}")
        heads_num = len(attention_weights[0])
    head_step = len(attention_weights[0]) // heads_num
    heads_ids = list(range(0, len(attention_weights[0]), head_step))
    columns_num = len(heads_ids) + 1

    image = base_image.permute(1, 2, 0).detach().cpu().numpy()
    patch_size = image.shape[0] // patches_per_axis

    # Choose patch to attend to
    rec_x = random.randint(0, patches_per_axis - 1)
    rec_y = random.randint(0, patches_per_axis - 1)
    attend_patch = rec_y * patches_per_axis + rec_x
    rect = patches.Rectangle(
        (rec_x * patch_size, rec_y * patch_size), patch_size, patch_size, linewidth=1, edgecolor="r", facecolor="none"
    )

    fig, axs = plt.subplots(rows_num, columns_num, figsize=(10, 10))
    axs[0, 0].imshow(image)
    axs[0, 0].add_patch(rect)

    for layer_id, row_id in zip(layers_ids, range(rows_num)):
        axs[row_id, 0].axis("off")
        for head_id, column_id in zip(heads_ids, range(1, columns_num)):
            ax = axs[row_id, column_id]
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if row_id == 0:
                ax.set_title(f"Head {head_id}.")
            if column_id == 1:
                ax.set_ylabel(f"Layer {layer_id}.")
            ax.imshow(image)
            attention = attention_weights[layer_id][head_id][attend_patch].reshape(patches_per_axis, patches_per_axis)
            attention = attention.detach().cpu().numpy()
            attention = np.repeat(np.repeat(attention, patch_size, axis=0), patch_size, axis=1)
            ax.imshow(attention, alpha=0.5, cmap="gray", interpolation="bilinear")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_cross_attention(
    base_image: torch.Tensor,
    caption_tokens: list[int],
    attention_weights: torch.Tensor,
    max_columns: int,
    patches_per_axis: int,
    decode_map: dict,
    show: bool = False,
):
    columns_number = max_columns
    if len(caption_tokens) < max_columns:
        print(f"Showing only {len(caption_tokens)} columns")
        columns_number = len(caption_tokens)

    rows_number = math.ceil(len(caption_tokens) / max_columns)

    image = base_image.permute(1, 2, 0).detach().cpu().numpy()
    patch_size = image.shape[0] // patches_per_axis

    fig, axs = plt.subplots(rows_number, columns_number, figsize=(10, 10))
    for row_id in range(rows_number):
        for column_id in range(columns_number):
            token_i = row_id * max_columns + column_id
            ax = axs[row_id, column_id] if rows_number > 1 else axs[column_id]
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if token_i < len(caption_tokens):
                ax.set_title(decode_map[caption_tokens[token_i]])
                ax.imshow(image)
                attention = attention_weights[token_i].reshape(patches_per_axis, patches_per_axis)
                attention = attention.detach().cpu().numpy()
                attention = np.repeat(np.repeat(attention, patch_size, axis=0), patch_size, axis=1)
                ax.imshow(attention, alpha=0.5, cmap="gray", interpolation="bilinear")
            else:
                ax.axis("off")
    fig.tight_layout()
    if show:
        plt.show()

    return fig


def visualize_candidates_graph(graph: CandidateGraph, show: bool = False):
    g, params = unravel_graph(graph)

    bbox_props = dict(boxstyle="round", fc="w")
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot", args="-Grankdir=LR")
    pos_labels = {key: (x, y-12) for key, (x, y) in pos.items()}

    fig, ax = plt.subplots()
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=params['edge_colors'], arrows=True, width=4, alpha=params['edge_alphas'])
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1000, node_shape='o', alpha=params['node_alphas'], node_color=params['node_colors'])
    nx.draw_networkx_labels(g, pos_labels, ax=ax, labels=params['labels'], bbox=bbox_props, font_size=8, verticalalignment='center_baseline', font_family='serif')
    ax.axis('off')
    fig.tight_layout()

    if show:
        plt.show()

    return fig, params['nodes']

