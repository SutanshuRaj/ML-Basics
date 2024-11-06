import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


def mean_squared_error(y_true, y_predicted):
	return np.mean((y_true - y_predicted) ** 2)


class LinearRegression():
	def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000) -> None:
		self.lr = learning_rate
		self.n_iters = n_iters
		self.weights = None
		self.bias = None


	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		# Gradiet Descent.
		for _ in range(self.n_iters):
			y_predicted = np.dot(X, self.weights) + self.bias

			# Compute the Gradient.
			dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
			db = (1 / n_samples) * np.sum(y_predicted - y)

			# Update the Parameters.
			self.weights -= self.lr * dw
			self.bias -= self.bias * db

	
	def predict(self, X):
		y_approx = np.dot(X, self.weights) + self.bias
		return y_approx



if __name__ == '__main__':
	
	X, y = datasets.make_regression(n_samples=1000, n_features=3, noise=20, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
	regressor.fit(X_train, y_train)
	predictions = regressor.predict(X_test)

	mse = mean_squared_error(y_test, predictions)
	print("MSE is: ", mse)

	cmap = plt.get_cmap("viridis")
	fig = plt.figure(figsize=(8, 6))
	plt.scatter(y_test, predictions, color=cmap(0.9), s=10, label="Prediction")
	plt.show()
	