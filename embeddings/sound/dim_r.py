# Helper functions to reduce dimensionality via several methods

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def reduce_PCA(x, y, n_components=3, title='Default Title', 
               xlabel='Component 1', ylabel='Component 2', zlabel='Component 3'):
    """
    Reduce dataset using PCA and plot the results.

    Parameters:
    - x: Input data.
    - y: Labels for the data.
    - n_components: Number of PCA components to retain.
    - title: Plot title.
    - xlabel, ylabel, zlabel: Labels for the axes.

    Returns:
    - fig: The figure object.
    - ax: The axes object.
    """

    if n_components > 3:
        return None, None

    pca = PCA(n_components)
    X_encoded_pca = pca.fit_transform(x)
    
    unique_labels = np.unique(y)
    color_palette = sns.color_palette("coolwarm", len(unique_labels))
    label_to_color = {label: color for label, color in zip(unique_labels, color_palette)}
    colors = [label_to_color[label] for label in y]  # Map colors to labels

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_encoded_pca[:, 0], X_encoded_pca[:, 1], color=colors, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        handles = [plt.Line2D([0], [0], marker="o", color="w", markersize=10, 
                            markerfacecolor=color) for color in color_palette]
        ax.legend(handles, unique_labels, title=title)
        
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_encoded_pca[:, 0], X_encoded_pca[:, 1], X_encoded_pca[:, 2], color=colors, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        
        handles = [plt.Line2D([0], [0], marker="o", color="w", markersize=10, 
                            markerfacecolor=color) for color in color_palette]
        ax.legend(handles, unique_labels, title=title)
        
        fig.tight_layout()
        fig.suptitle(title)
        
        handles = [plt.Line2D([0], [0], marker="o", color="w", markersize=10, 
                            markerfacecolor=color) for color in color_palette]
        fig.legend(handles, unique_labels, title=title, loc='upper right')
        
    return fig, ax

def reduce_tSNE(x, y, n_components=3, perplexity=30, random_state=42, 
               title='Default Title', xlabel='t-SNE Component 1', ylabel='t-SNE Component 2', zlabel='t-SNE Component 3'):
    """
    Reduce dataset using t-SNE and plot the results.

    Parameters:
    - x: Input data.
    - y: Labels for the data.
    - n_components: Number of t-SNE components to retain.
    - perplexity: Perplexity parameter for t-SNE.
    - random_state: Random state for t-SNE.
    - title: Plot title.
    - xlabel, ylabel, zlabel: Labels for the axes.

    Returns:
    - fig: The figure object.
    - ax: The axes object.
    """

    if n_components > 3:
        n_components = 3

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, learning_rate=150, max_iter=1000)
    X_encoded_tsne = tsne.fit_transform(x)
    
    unique_labels = np.unique(y)
    color_palette = sns.color_palette("husl", len(unique_labels))  # Use a distinct color palette
    label_to_color = {label: color for label, color in zip(unique_labels, color_palette)}

    if n_components == 2:
        # For 2 components, create a simple scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for label in unique_labels:
            label_mask = y == label
            ax.scatter(X_encoded_tsne[label_mask, 0], X_encoded_tsne[label_mask, 1], 
                       color=label_to_color[label], label=label, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="Labels")
        
    elif n_components == 3:
        # For 3 components, create a 3D scatter plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for label in unique_labels:
            label_mask = y == label
            ax.scatter(X_encoded_tsne[label_mask, 0], X_encoded_tsne[label_mask, 1], X_encoded_tsne[label_mask, 2], 
                       color=label_to_color[label], label=label, alpha=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.legend(title="Labels")
        
        fig.tight_layout()
        fig.suptitle(title)
        
    return fig, ax

def reduce_handcrafted():
    # Load features and labels
    all_features = np.load('processed_sound_files/hc_audio_features.npy')
    file_names = np.load('processed_sound_files/hc_audio_labels.npy')

    # Encode labels
    label_encoder = LabelEncoder()
    file_name_labels = label_encoder.fit_transform(file_names)
    num_classes = len(np.unique(file_name_labels))

    # Use tab20 colormap for more distinct colors
    cmap = plt.cm.get_cmap("tab20", num_classes)
    colors = cmap(file_name_labels % cmap.N)  # Ensure colors wrap around if more than cmap.N

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = all_features[:, 0]  # Spectral Centroid
    y = all_features[:, 1]  # Zero Crossing Rate
    z = all_features[:, 2]  # RMS Energy

    # Scatter plot
    ax.scatter(x, y, z, c=colors)

    ax.set_xlabel('Spectral Centroid')
    ax.set_ylabel('Spectral Flux')
    ax.set_zlabel('RMS Energy')
    ax.set_title('Audio Features Scatterplot')

    # Legend setup
    unique_labels = np.unique(file_name_labels)
    legend_labels = label_encoder.inverse_transform(unique_labels)
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color='w', label=label,
                markerfacecolor=cmap(i % cmap.N), markersize=10) 
        for i, label in enumerate(legend_labels)
    ]

    ax.legend(handles=legend_patches, title="Instruments", loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X_encoded = np.load('model_outputs/autoencoder_output.npy')
    y = np.load('model_outputs/autoencoder_labels.npy')

    fig, axs = reduce_tSNE(X_encoded, y, n_components=2, perplexity=25, title="tSNE Reduced Dimensionality Map")
    # fig, axs = reduce_PCA(X_encoded, y, n_components=3, title="PCA Reduced Dimensionality Map")
    # fig, axs = reduce_PCA(X_encoded, y, n_components=2, title='PCA reduced plot of Centroid vs. Flux vs. RMS NRG')

    plt.tight_layout()
    plt.show()