import numpy as np
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split


cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def accuracy(y_true, y_predicted):
    accuracy = np.sum(y_true == y_predicted) / len(y_true)
    return accuracy


class kNN:
    def __init__(self, k:int = 3) -> None:
        self.k = k


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        y_predicted = [self._predict(x) for x in X]
        return np.array(y_predicted)


    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort Distance and Return Indices.
        k_idx = np.argsort(distances)[: self.k]

        # Extract the Labels.
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        majority_vote = Counter(k_neighbor_labels).most_common(1)
        return majority_vote[0][0]


if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    k = 7
    clf = kNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("kNN Classification Accuracy: ", accuracy(y_test, predictions))
