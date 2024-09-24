from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

# Example: Assume you have sequences embedded as vectors
# sequences: a list of sequence embeddings (numpy arrays)
sequences = np.array([
    [0.5, 0.2, 0.1],  # sequence 1 embedding
    [0.6, 0.1, 0.3],  # sequence 2 embedding
    [0.1, 0.9, 0.8],  # sequence 3 embedding
    # ... more embeddings
])

# Apply dimensionality reduction (optional)
pca = PCA(n_components=2)  # Reduce to 2D for visualization
reduced_sequences = pca.fit_transform(sequences)

# Clustering using K-Means
kmeans = KMeans(n_clusters=3, random_state=0)  # Set number of clusters
clusters = kmeans.fit_predict(reduced_sequences)

# Visualize the clusters (optional)
import matplotlib.pyplot as plt
plt.scatter(reduced_sequences[:, 0], reduced_sequences[:, 1], c=clusters, cmap='viridis')
plt.title('Sequence Clustering')
plt.show()
