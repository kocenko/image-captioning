import numpy as np
from scipy.special import softmax
from matplotlib import pyplot as plt


# For reproducibility
np.random.seed(42)

# Word embeddings (encoder's output)
words_number = 5
embeddings_size = 3
words = np.random.randint(0, 1 + 1, (words_number, embeddings_size))

# Weights for queries, keys and values
weights_range = 3
queries_weights = np.random.randint(weights_range, size=(embeddings_size, embeddings_size))
keys_weights = np.random.randint(weights_range, size=(embeddings_size, embeddings_size))
values_weights = np.random.randint(weights_range, size=(embeddings_size, embeddings_size))

# Queries, keys and values
queries = words @ queries_weights
keys = words @ keys_weights
values = words @ values_weights

# Scoring queries against keys
scores = queries @ keys.T

# Weights
weights = softmax(scores / keys.shape[1] ** .5, axis=1)

# Attention matrix and map
attention = weights @ values

print(attention)
plt.imshow(attention)
plt.ylabel('Word number')
plt.xlabel('Embeddings')
plt.show()
