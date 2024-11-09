import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(42)

def euclidean_distance(x, y):
	return np.sqrt(np.sum((x - y) ** 2))


class KMeans():
	def __init__(self, K=5, max_iters=100, plot_steps=False) -> None:
		self.K = K
		self.max_iters = max_iters
		self.plot_steps = plot_steps
		self.clusters = [[] for _ in range(self.K)]
		self.centroids = []


	def predict(self, X):
		self.X = X
		self.n_samples, self.n_features = X.shape

		# Initialize the Centroids.
		random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
		self.centroids = [self.X[idx] for idx in random_sample_idxs]

		# Optimization: update Clusters and Centroids.
		for _ in range(self.max_iters):
			self.clusters = self._create_clusters(self.centroids)
			centroids_old = self.centroids
			self.centroids = self._get_centroids(self.clusters)

			if self._is_converged(centroids_old, self.centroids):
				break

		# Return Cluster Label.
		return self._get_cluster_labels(self.clusters)


	def _create_clusters(self, centroids):
		clusters = [[] for _ in range(self.K)]
		for idx, sample in enumerate(self.X):
			centroid_idx = self._closest_centroid(sample, centroids)
			clusters[centroid_idx].append(idx)
		return clusters


	def _closest_centroid(self, sample, centroids):
		distances = [euclidean_distance(sample, cent) for cent in centroids]
		closest_idx = np.argmin(distances)
		return closest_idx


	def _get_centroids(self, clusters):
		centroids = np.zeros((self.K, self.n_features))
		for cluster_idx, cluster in enumerate(clusters):
			cluster_mean = np.mean(self.X[cluster], axis=0)
			centroids[cluster_idx] = cluster_mean
		return centroids


	def _is_converged(self, centroids_old, centroids):
		distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
		return sum(distances) == 0


	def _get_cluster_labels(self, clusters):
		labels = np.empty(self.n_samples)
		for cluster_idx, cluster in enumerate(clusters):
			for sample_idx in cluster:
				labels[sample_idx] = cluster_idx
		return labels


	def plot(self):
		fig, ax = plt.subplots(figsize=(12, 8))

		for i, index in enumerate(self.clusters):
			point = self.X[index].T
			ax.scatter(*point)

		for point in self.centroids:
			ax.scatter(*point, marker="x", color="black", linewidth=2)

		plt.show()



if __name__ == '__main__':
	X, y = datasets.make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
	clusters = len(np.unique(y))
	print(clusters)

	unsupervised = KMeans(K=clusters, max_iters=150, plot_steps=False)
	y_predict = unsupervised.predict(X)

	unsupervised.plot()