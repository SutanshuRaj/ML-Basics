import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets


def accuracy(y_true, y_predicted):
    accuracy = np.sum(y_true == y_predicted) / len(y_true)
    return accuracy


class LogisticRegression():
	def __init__(self, learning_rate:float = 0.001, n_iters:int = 1000) -> None:
		self.lr = learning_rate
		self.n_iters = n_iters
		self.weights = None 
		self.bias = None 

	
	def fit(self, X, y):
		n_samples, n_featues = X.shape
		self.weights = np.zeros(n_featues)
		self.bias = 0

		# Gradient Descent.
		for _ in range(self.n_iters): 
			linear_model = np.dot(X, self.weights) + self.bias
			y_predicted = self._sigmoid(linear_model)

			# Compute the Gradient.
			dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
			db = (1 / n_samples) * np.sum(y_predicted - y)

			# Update the Parameters.
			self.weights -= self.lr * dw
			self.bias -= self.bias * db


	def predict(self, X):
		linear_model = np.dot(X, self.weights) + self.bias
		y_predicted = self._sigmoid(linear_model)
		y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
		return np.array(y_predicted_cls)


	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))



if __name__ == '__main__':
	archive = datasets.load_breast_cancer()
	X, y = archive.data, archive.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	clf = LogisticRegression(learning_rate=0.0001, n_iters=10000)
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)

	print("Accuracy of Logistic Regression is: ", accuracy(y_test, predictions))