import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Instrument mapping (same as your code)
instrument_map = {
    "Tp": "Trumpet", "Tbn": "Trombone", "Hn": "Horn", "BTb": "Bass Trombone",
    "Vc": "Violincello", "Vn": "Violin", "Va": "Viola", "Cb": "Contrabass",
    "Hp": "Harp", "Gtr": "Guitar", "Acc": "Accordion", "ASax": "Alto Sax",
    "Ob": "Oboe", "Fl": "Flute", "ClBb": "Clarinet", "Bn": "Bassoon"
}

# Load features and labels
all_features = np.load('audio_features.npy')
file_names = np.load('audio_labels.npy')

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
scatter = ax.scatter(x, y, z, c=colors)

ax.set_xlabel('Spectral Centroid')
ax.set_ylabel('Zero Crossing Rate')
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
