import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, l2_penalty=0.0, batch_size=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / m

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
            if self.batch_size is None or self.batch_size >= num_samples:
                X_batch, y_batch = X, y_encoded
            else:
                idx = np.random.choice(num_samples, self.batch_size, replace=False)
                X_batch, y_batch = X[idx], y_encoded[idx]
            
            logits = np.dot(X_batch, self.weights) + self.bias
            probabilities = self.softmax(logits)
            
            loss = self.cross_entropy_loss(y_batch, probabilities)
            reg_loss = 0.5 * self.l2_penalty * np.sum(self.weights**2)
            total_loss = loss + reg_loss
            self.loss_history.append(total_loss)
            
            y_pred = np.argmax(probabilities, axis=1)
            accuracy = np.mean(np.argmax(y_batch, axis=1) == y_pred)
            self.accuracy_history.append(accuracy)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.4f}")
            
            dw = (1/len(X_batch)) * np.dot(X_batch.T, (probabilities - y_batch)) + (self.l2_penalty * self.weights)
            db = (1/len(X_batch)) * np.sum(probabilities - y_batch, axis=0, keepdims=True)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(logits)
        return np.argmax(probabilities, axis=1)

    def plot_training_curves(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history, label='Accuracy', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        
        plt.show()

    def macro_precision(self, y_true, y_pred, num_classes):
        precision_per_class = []
        for c in range(num_classes):
            true_positive = np.sum((y_true == c) & (y_pred == c))
            false_positive = np.sum((y_true != c) & (y_pred == c))
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
            precision_per_class.append(precision)
        return precision_per_class

    def macro_recall(self, y_true, y_pred, num_classes):
        recall_per_class = []
        for c in range(num_classes):
            true_positive = np.sum((y_true == c) & (y_pred == c))
            false_negative = np.sum((y_true == c) & (y_pred != c))
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
            recall_per_class.append(recall)
        return recall_per_class

    def macro_f1_score(self, precision_scores, recall_scores):
        f1_scores = []
        for precision, recall in zip(precision_scores, recall_scores):
            if (precision + recall) != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
        return f1_scores

    def macro_avg(self, metric_scores):
        return np.mean(metric_scores)

    def weighted_avg(self, metric_scores, y_true):
        class_counts = np.bincount(y_true)
        total_samples = len(y_true)
        weighted_avg = np.sum(np.array(metric_scores) * class_counts) / total_samples
        return weighted_avg

