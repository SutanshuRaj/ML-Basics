import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets


class LDA():
	def __init__(self, n_dims) -> None:
		self.n_dims = n_dims
		self.linear_discriminants = None


	def fit(self, X, y):
		n_features = X.shape[1]
		class_labels = np.unique(y)

		# Calculate Scatter Matrix.
		mean_net = np.mean(X, axis=0)
		S_w = np.zeros((n_features, n_features))
		S_b = np.zeros((n_features, n_features))		

		for c in class_labels:
			X_c = X[y == c]
			mean_c = np.mean(X_c, axis=0)

			S_w += (X_c - mean_c).T.dot(X_c - mean_c)

			num_c = X_c.shape[0]
			mean_diff = (mean_c - mean_net).reshape(n_features, 1)
			S_b += num_c * (mean_diff).dot(mean_diff.T)

		loss_func = np.linalg.inv(S_w).dot(S_b)

		# Calculate Eigen Values and Vectors.
		eigen_values, eigen_vectors = np.linalg.eig(loss_func)
		eigen_vectors = eigen_vectors.T
		idx = np.argsort(abs(eigen_values))[::-1]
		eigen_values = eigen_values[idx]
		eigen_vectors = eigen_vectors[idx]

		self.linear_discriminants = eigen_vectors[0:self.n_dims]


	def transform(self, X):
		return np.dot(X, self.linear_discriminants.T)



if __name__ == "__main__":
	archive = datasets.load_iris()
	X, y = archive.data, archive.target

	lda = LDA(2)
	lda.fit(X, y)
	X_projected = lda.transform(X)

	print("Shape of Features: ", X.shape)
	print("Shape after LDA: ", X_projected.shape)

	x1 = X_projected[ : , 0]
	x2 = X_projected[ : , 1]
	
	plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8)
	plt.xlabel("Principle Component 1.")
	plt.ylabel("Principle Component 2.")
	plt.colorbar()
	plt.show()