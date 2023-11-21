import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch


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


def plot_self_attention(
    base_image: torch.Tensor,
    attention_weights: list[list[torch.Tensor]],
    layers_num: int,
    heads_num: int,
    patch_size: int,
):
    assert layers_num <= len(attention_weights), f"Cannot visualize more layers than {len(attention_weights)}"
    layer_step = len(attention_weights) // layers_num
    layers_to_plot = attention_weights[0::layer_step]
    rows_num = len(layers_to_plot)

    assert heads_num <= len(layers_to_plot[0]), f"Cannot visualize more heads than {len(layers_to_plot[0])}"
    head_step = len(layers_to_plot[0]) // heads_num
    layers_to_plot = [layer[0::head_step] for layer in layers_to_plot]
    columns_num = len(layers_to_plot[0]) + 1

    fig, axs = plt.subplots(rows_num, columns_num, figsize=(10, 10))
    axs[0, 0].imshow(base_image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
