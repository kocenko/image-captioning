import os
import matplotlib.pyplot as plt
import numpy as np


def plot_logs(name: str, folder: str = "./numpy_logs", batch_num: int = 0):
    all_files = os.listdir(folder)
    all_numpy = [file.split(".", maxsplit=1)[0] for file in all_files if file.split(".", maxsplit=1)[1] == "npy"]
    all_match = [os.path.join(folder, f"{file}.npy") for file in all_numpy if file.split("_", maxsplit=1)[1] == name]
    assert len(all_match) == 2, "Expected two files with the given name"

    array1 = np.load(all_match[0])
    array2 = np.load(all_match[1])

    assert (
        array1.shape == array2.shape
    ), f"Shapes {array1.shape} of {all_match[0]} and {array2.shape} of {all_match[1]} do not match"
    assert batch_num < array1.shape[0], 'Wrong batches num'

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
                ax.set_ylabel(f"Head {i}")
                ax.imshow(array1[i])
            if j == 1:
                ax.imshow(array2[i])
            if i == 0:
                ax.set_title(f"{all_match[j]}")

    plt.tight_layout()
    plt.show()
