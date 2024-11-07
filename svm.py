import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt


class SVM():
	def __init__(self, learning_rate:float = 0.001, lambda_param:float = 0.01, n_iters:int = 1000) -> None:
		self.lr = learning_rate
		self.n_iters = n_iters
		self.lambda_param = lambda_param
		self.weights = None
		self.bias = None


	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		y_ = np.where(y <= 0, -1, 1)

		for _ in range(self.n_iters):
			for idx, x_i in enumerate(X):

				condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1

				# Gradient Descent Params.
				if condition:
					self.weights -= self.lr * (2 * self.lambda_param * self.weights)
				else:
					self.weights -= self.lr * ((2 * self.lambda_param * self.weights) - np.dot(x_i, y_[idx]))
					self.bias -= self.lr * y_[idx]


	def predict(self, X):
		y_approx = np.dot(X, self.weights) - self.bias
		return np.sign(y_approx)


if __name__ == '__main__':
	X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.25, random_state=42)
	y = np.where(y == 0, -1, 1)

	clf = SVM()
	clf.fit(X, y)
	predictions = clf.predict(X)
	