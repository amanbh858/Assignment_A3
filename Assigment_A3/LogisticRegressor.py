from sklearn.metrics import classification_report
import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, l2_penalty=0.0,iterations=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.iterations=iterations
        self.weights = None
        self.bias = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability improvement
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot_encode(self, y, num_classes):
        m = y.shape[0]
        y_encoded = np.zeros((m, num_classes))
        y_encoded[np.arange(m), y] = 1
        return y_encoded

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros((1, num_classes))
        y_encoded = self.one_hot_encode(y, num_classes)

        for _ in range(self.epochs):
            logits = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(logits)
            
            dw = (1 / num_samples) * np.dot(X.T, (probabilities - y_encoded)) + (self.l2_penalty * self.weights)
            db = (1 / num_samples) * np.sum(probabilities - y_encoded, axis=0, keepdims=True)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict1(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(logits)
        return np.argmax(probabilities, axis=1) 

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def macro_precision(self, y_true, y_pred, num_classes):
        precision_scores = []
        for c in range(num_classes):
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            precision_scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        return precision_scores

    def macro_recall(self, y_true, y_pred, num_classes):
        recall_scores = []
        for c in range(num_classes):
            tp = np.sum((y_pred == c) & (y_true == c))
            fn = np.sum((y_pred != c) & (y_true == c))
            recall_scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        return recall_scores

    def macro_f1_score(self, precision, recall):
        return [
            (2 * p * r) / (p + r) if (p + r) > 0 else 0 
            for p, r in zip(precision, recall)
        ]
    
    def macro_avg(self, scores):
        return np.mean(scores)

    def weighted_avg(self, scores, y_true):
        class_counts = np.bincount(y_true)
        total = len(y_true)
        return np.sum((class_counts / total) * scores)

