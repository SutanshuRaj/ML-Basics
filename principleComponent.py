import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt


class PCA():
	def __init__(self, n_dims) -> None:
		self.n_dims = n_dims
		self.dims = None
		self.mean = None


	def fit(self, X):
		# Mean Cetering.
		self.mean = np.mean(X, axis=0)
		X -= self.mean

		# Covariance.
		cov = np.cov(X.T)

		# Sort Eigen values.
		eigen_values, eigen_vectors = np.linalg.eig(cov)

		eigen_vectors = eigen_vectors.T
		idx = np.argsort(eigen_values)[::-1]
		eigen_values = eigen_values[idx]
		eigen_vectors = eigen_vectors[idx]

		self.dims = eigen_vectors[0: self.n_dims]


	def transform(self, X):
		# Project the Data.
		X -= self.mean
		return np.dot(X, self.dims.T)



if __name__ == '__main__':
	archive = datasets.load_digits()
	X, y = archive.data, archive.target

	pca = PCA(2)
	pca.fit(X)
	X_projected = pca.transform(X)

	print("Shape of Features: ", X.shape)
	print("Shape after PCA: ", X_projected.shape)

	x1 = X_projected[ : , 0]
	x2 = X_projected[ : , 1]
	
	plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8)
	plt.xlabel("Principle Component 1.")
	plt.ylabel("Principle Component 2.")
	plt.colorbar()
	plt.show()