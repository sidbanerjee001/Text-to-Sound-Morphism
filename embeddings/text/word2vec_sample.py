# Imports a pretrained word2vec model, selects the top 500 words, and reduces the dimensionality to 3D via tSNE

import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import umap

def normalize_data(data):
    """
    Returns a new array where each row is normalized by its max value.
    value = value / max_value
    """
    data = np.array(data)
    normalized = np.empty_like(data, dtype=float)

    for i in range(data.shape[0]):
        mx = np.max(data[i])
        if mx != 0:
            normalized[i] = data[i] / mx
        else:
            normalized[i] = data[i]

    return normalized

print("Loading Word2Vec model...")
model = api.load("word2vec-google-news-300")

words = list(model.key_to_index.keys())[:10000]
vectors = normalize_data(np.array([model[word] for word in words]))

# print("Running t-SNE...")
# tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
# vectors_3d = tsne.fit_transform(vectors)

print("Running UMAP...")
reducer = umap.UMAP(n_components=3, random_state=42)
vectors_3d = reducer.fit_transform(vectors)

np.savez("embedding_outputs/umap10k_word2vec_3d.npz", vectors=vectors_3d, words=np.array(words))