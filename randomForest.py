import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from decisionTree import DecisionTree, accuracy


def bootstrap_sample(X, y):
	n_samples = X.shape[0]
	idx = np.random.choice(n_samples, size=n_samples, replace=True)
	return X[idx], y[idx]


def most_common_label(y):
	counter = Counter(y)
	most_common = counter.most_common(1)[0][0]
	return most_common



class RandomForest():
	def __init__(self, n_trees:int=100, min_samples_split:int=2, max_depth:int=100, n_feats=None) -> None:
		self.n_trees = n_trees
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.n_feats = n_feats
		self.trees = []


	def fit(self, X, y):
		self.trees = []

		for _ in range(self.n_trees):
			tree = DecisionTree(min_samples_split=self.min_samples_split,
								max_depth=self.max_depth, n_feats=self.n_feats)
			X_sample, y_sample = bootstrap_sample(X, y)
			tree.fit(X_sample, y_sample)
			self.trees.append(tree)


	def predict(self, X):
		tree_predict = np.array([tree.predict(X) for tree in self.trees])
		tree_predict = np.swapaxes(tree_predict, 0, 1)
		y_predict = [most_common_label(tree_pred) for tree_pred in tree_predict]
		return np.array(y_predict)



if __name__ == '__main__':
	archive = datasets.load_breast_cancer()
	X, y = archive.data, archive.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	clf = RandomForest(n_trees=3, max_depth=10)
	clf.fit(X_train, y_train)
	predictions = clf.predict(X_test)

	print('Accuracy for Random Forest is: ', accuracy(y_test, predictions))
