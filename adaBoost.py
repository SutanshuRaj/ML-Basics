import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decisionTree import accuracy

class DecisionRoot():
	def __init__(self) -> None:
		self.polarity = 1
		self.feature_idx = None
		self.threshold = None
		self.alpha = None

	def predict(self, X):
		n_samples = X.shape[0]
		X_column = X[ : , self.feature_idx]

		predictions = np.ones(n_samples)
		if self.polarity == 1:
			predictions[X_column < self.threshold] = -1
		else:
			predictions[X_column > self.threshold] = -1

		return predictions


class AdaBoost():
	def __init__(self, n_clf=5) -> None:
		self.n_clf = n_clf
		self.clfs = []

	def fit(self, X, y):
		n_samples, n_features = X.shape

		# Initialize the Weights.
		weights = np.full(n_samples, (1/n_samples))
		self.clfs = []

		# Iterate through Classifiers.
		for _ in range(self.n_clf):
			clf = DecisionRoot()

			min_error = float('Inf')

			# Greedy Search, to find best Threshold and Features.
			for feature_i in range(n_features):
				X_column = X[ : , feature_i]
				thresholds = np.unique(X_column)

				for threshold in thresholds:
					polar = 1
					predictions = np.ones(n_samples)
					predictions[X_column < threshold] = -1

					misclassified  = weights[y != predictions]
					error = np.sum(misclassified)

					if error > 0.5:
						error = 1 - error
						polar = -1

					# Save Best Configuration.
					if error < min_error:
						min_error = error
						clf.polarity = polar
						clf.threshold = threshold
						clf.feature_idx = feature_i

			EPS = 1e-10
			clf.alpha = 0.5 * np.log((1-error) / (error + EPS))

			predictions = clf.predict(X)
			weights *= np.exp(-clf.alpha * y * predictions)
			weights /= np.sum(weights)

			self.clfs.append(clf)



	def predict(self, X):
		clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
		y_predict = np.sum(clf_preds, axis=0)
		y_predict = np.sign(y_predict)
		return y_predict



if __name__ == '__main__':
	archive = datasets.load_breast_cancer()
	X, y = archive.data, archive.target

	y[y==0] = -1
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	clf = AdaBoost(n_clf=5)
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)

	print("Accuracy of AdaBoost is: ", accuracy(y_test, predictions))
