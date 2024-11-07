import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_true, y_predicted):
    accuracy = np.sum(y_true == y_predicted) / len(y_true)
    return accuracy


class Perceptron():
	def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000) -> None:
		self.lr = learning_rate
		self.n_iters = n_iters
		self.weights = None 
		self.bias = None
		self.activation_func = self._unit_step_func

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iters):
			for idx, x_i in enumerate(X):
				linear_model = np.dot(x_i, self.weights) + self.bias
				y_predicted = self.activation_func(linear_model)

				# Perceptron Update Rule.
				delta = self.lr * (y[idx] - y_predicted)
				self.weights += delta * x_i
				self.bias += delta


	def predict(self, X):
		linear_approx = np.dot(X, self.weights) + self.bias
		y_predicted = self.activation_func(linear_approx)
		return y_predicted


	# Define Activation Function.
	def _unit_step_func(self, x):
		return np.where(x >= 0, 1, 0)


if __name__ == '__main__':

	X, y = datasets.make_blobs(n_samples=1000, n_features=3, centers=2, cluster_std=1.35, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	clf = Perceptron(learning_rate=0.01, n_iters=1000)
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)

	print("Perceptron Classification Accuracy is: ", accuracy(y_test, predictions))

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

	x0_1 = np.amin(X_train[:, 0])
	x0_2 = np.amax(X_train[:, 0])

	x1_1 = (-clf.weights[0] * x0_1 - clf.bias) / clf.weights[1]
	x1_2 = (-clf.weights[0] * x0_2 - clf.bias) / clf.weights[1]

	ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

	ymin = np.amin(X_train[:, 1])
	ymax = np.amax(X_train[:, 1])
	ax.set_ylim([ymin - 3, ymax + 3])

	plt.show()