import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import umap

# Cargar todos los .pkl
all_embeddings = []
all_labels = []
all_sentences = []

for file in os.listdir('./embeddings/'):
    if file.endswith(".pkl") and "post" in file:
        class_label = file.split('_')[0]  # nombre del archivo sin extensión
        with open(os.path.join('./embeddings/', file), "rb") as f:
            sentences, embeddings = pickle.load(f)
            all_embeddings.append(embeddings)
            all_labels.extend([class_label] * len(embeddings))
            all_sentences.extend(sentences)

# Unir todo
all_embeddings = np.vstack(all_embeddings)

# UMAP en 3D
reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
reduced = reducer.fit_transform(all_embeddings)

# Asignar colores
unique_labels = sorted(set(all_labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for label in unique_labels:
    idxs = [i for i, l in enumerate(all_labels) if l == label]
    coords = reduced[idxs]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               label=label, color=label_color_map[label], alpha=0.7)

ax.set_title("bdi-mpnet-base")
# ax.set_xlabel("UMAP-1")
# ax.set_ylabel("UMAP-2")
# ax.set_zlabel("UMAP-3")
ax.legend()
plt.tight_layout()
plt.show()