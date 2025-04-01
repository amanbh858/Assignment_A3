import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, l2_penalty=0.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / m  # Added epsilon for numerical stability

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

        for epoch in range(self.epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(logits)
            
            # Calculate metrics
            loss = self.cross_entropy_loss(y_encoded, probabilities)
            reg_loss = 0.5 * self.l2_penalty * np.sum(self.weights**2)
            total_loss = loss + reg_loss
            self.loss_history.append(total_loss)
            
            y_pred = np.argmax(probabilities, axis=1)
            accuracy = np.mean(y == y_pred)
            self.accuracy_history.append(accuracy)
            
            # Print training progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.4f}")
            
            # Backward pass
            dw = (1/num_samples) * np.dot(X.T, (probabilities - y_encoded)) + (self.l2_penalty * self.weights)
            db = (1/num_samples) * np.sum(probabilities - y_encoded, axis=0, keepdims=True)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
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

    def plot_training_curves(self):
        plt.figure(figsize=(12, 5))
        