import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class CustomRegression:
    """
    A custom regression class that supports various optimization methods, regularization, and polynomial features.
    """
    
    def __init__(self, regularization: float = None, learning_rate: float = 0.01, optimization_method: str = 'mini_batch',
                 weight_initialization: str = 'zeros', batch_size: int = 64, num_epochs: int = 100, momentum: float = None,
                 polynomial_degree: int = None, cross_validation_folds: int = 5, apply_log_transform: bool = False, use_mlflow: bool = False):
        """
        Initialize the regression model with various hyperparameters.
        
        Args:
            regularization (float): Regularization strength (None, L1, or L2).
            learning_rate (float): Learning rate for gradient descent.
            optimization_method (str): Optimization method ('mini_batch', 'stochastic', 'batch').
            weight_initialization (str): Weight initialization method ('zeros', 'normal', 'xavier').
            batch_size (int): Batch size for mini-batch gradient descent.
            num_epochs (int): Number of epochs for training.
            momentum (float): Momentum coefficient for gradient descent.
            polynomial_degree (int): Degree of polynomial features (None for no polynomial features).
            cross_validation_folds (int): Number of cross-validation folds.
            apply_log_transform (bool): Whether to apply log transformation to the target variable.
            use_mlflow (bool): Whether to use MLflow for tracking experiments.
        """
        self.learning_rate = learning_rate
        self.optimization_method = optimization_method
        self.weight_initialization = weight_initialization
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.regularization = regularization
        self.momentum = momentum
        self.use_mlflow = use_mlflow
        self.polynomial_degree = polynomial_degree
        self.cross_validation_folds = cross_validation_folds
        self.weight_decay = 1e-5  # L2 regularization
        self.apply_log_transform = apply_log_transform

        # Validate optimization method and weight initialization
        valid_methods = ['mini_batch', 'stochastic', 'batch']
        valid_weights = ['normal', 'xavier', 'zeros']
        
        if self.optimization_method not in valid_methods:
            raise ValueError(f'optimization_method must be in {valid_methods}')
        if self.weight_initialization not in valid_weights:
            raise ValueError(f'weight_initialization must be in {valid_weights}')

    def train_model(self, X, y):
        """
        Train the model on the provided data.
        
        Args:
            X (pd.DataFrame or np.array): Feature matrix.
            y (pd.Series or np.array): Target vector.
        """
        # Convert inputs to numpy arrays if they are pandas objects
        X = self._convert_to_numpy(X)
        y = self._convert_to_numpy(y)

        # Apply log transformation if specified
        if self.apply_log_transform:
            y = np.log1p(y)

        # Add polynomial features if specified
        if self.polynomial_degree is not None:
            X = self._generate_polynomial_features(X)

        # Add intercept term to the feature matrix
        X = self._add_bias_term(X)

        # Initialize weights
        self.weights = self._initialize_weights(X.shape[1])

        # Perform K-Fold cross-validation
        self._perform_kfold_validation(X, y)

    def _convert_to_numpy(self, data):
        """Convert pandas DataFrame or Series to numpy array."""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.astype('float').values
        return np.array(data).astype('float')

    def _add_bias_term(self, X):
        """Add a column of ones to the feature matrix for the intercept term."""
        intercept = np.ones((X.shape[0], 1)).astype('float')
        return np.concatenate((intercept, X), axis=1)

    def _perform_kfold_validation(self, X, y):
        """Perform K-Fold cross-validation."""
        kfold = KFold(n_splits=self.cross_validation_folds, shuffle=True, random_state=42)
        self.train_accuracy_history = []
        self.train_loss_history = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            self.weights = self._initialize_weights(X.shape[1])

            x_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            x_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            self.best_loss = np.inf
            epochs_without_improvement = 0

            for epoch in range(self.num_epochs):
                perm_idx = np.random.permutation(len(x_train_fold))
                x_train_fold = x_train_fold[perm_idx]
                y_train_fold = y_train_fold[perm_idx]

                if self.optimization_method == 'mini_batch':
                    for i in range(0, len(x_train_fold), self.batch_size):
                        batch_X = x_train_fold[i:i + self.batch_size]
                        batch_y = y_train_fold[i:i + self.batch_size]
                        train_loss, train_accuracy = self._update_weights(batch_X, batch_y)
                elif self.optimization_method == 'stochastic':
                    for i in range(x_train_fold.shape[0]):
                        batch_X = x_train_fold[i:i + 1]
                        batch_y = y_train_fold[i:i + 1]
                        train_loss, train_accuracy = self._update_weights(batch_X, batch_y)
                else:
                    train_loss, train_accuracy = self._update_weights(x_train_fold, y_train_fold)

                self.train_accuracy_history.append(train_accuracy)
                self.train_loss_history.append(train_loss)

                y_pred_val = self._make_prediction(x_val_fold, is_training=True)
                val_loss, val_accuracy = self._calculate_mse(y_val_fold, y_pred_val), self._calculate_r2(y_val_fold, y_pred_val)

                if np.isclose(self.best_loss, val_loss, atol=1e-3):
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= 4:  # Early stopping patience
                        print('Early stopping triggered')
                        break
                else:
                    epochs_without_improvement = 0

                self.best_loss = val_loss

            print(f"Fold {fold} --> Train Loss: {np.mean(self.train_loss_history):.3f} | "
                  f"Train Accuracy: {np.mean(self.train_accuracy_history):.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

    def _update_weights(self, X, y):
        """Update model weights using gradient descent."""
        y_pred = self._make_prediction(X, is_training=True)
        m = X.shape[0]

        # Compute gradient
        gradient = (1 / m) * np.dot(X.T, (y_pred - y).reshape(-1, 1))
        gradient += self.weight_decay * self.weights.reshape(-1, 1)  # Add L2 regularization
        gradient = np.clip(gradient, -5.0, 5.0)  # Clip gradients to avoid exploding gradients

        # Update weights with momentum if specified
        if self.momentum is not None:
            if not hasattr(self, 'velocity'):
                self.velocity = np.zeros_like(self.weights)
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient.flatten()
            self.weights += self.velocity.flatten()
        else:
            self.weights -= self.learning_rate * gradient.flatten()

        # Handle NaN or Inf weights
        if np.isnan(self.weights).any() or np.isinf(self.weights).any():
            print("Warning: NaN or Inf detected in weights!")
            self.weights = np.where(np.isnan(self.weights) | np.isinf(self.weights), np.random.randn(*self.weights.shape) * 0.01, self.weights)

        return self._calculate_mse(y, y_pred), self._calculate_r2(y, y_pred)

    def _make_prediction(self, X, is_training=False):
        """Predict the target values for the given feature matrix."""
        X = self._convert_to_numpy(X)

        if not is_training and self.polynomial_degree is not None:
            X = self._generate_polynomial_features(X)

        if X.shape[1] == self.weights.shape[0] - 1:  # Check if bias is missing
            X = self._add_bias_term(X)

        return np.dot(X, self.weights.reshape(-1, 1)).flatten()

    def _calculate_mse(self, y_true, y_pred):
        """Compute Mean Squared Error."""
        return np.sum((y_pred - y_true) ** 2) / y_true.shape[0]

    def _calculate_r2(self, y_true, y_pred):
        """Compute R-squared score."""
        rss = np.sum((y_true - y_pred) ** 2)
        tss = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (rss / tss)

    def _initialize_weights(self, num_features):
        """Initialize weights based on the specified method."""
        if self.weight_initialization == 'zeros':
            return np.zeros(num_features).astype('float')
        elif self.weight_initialization == 'xavier':
            limit = 1 / np.sqrt(num_features)
            return np.random.uniform(-limit, limit, size=num_features).astype('float')
        else:
            return np.random.randn(num_features).astype('float') * 0.01

    def _generate_polynomial_features(self, X):
        """Generate polynomial features."""
        if self.polynomial_degree is None:
            return X

        X = np.array(X, dtype='float')
        poly_features = [X]

        for degree in range(2, self.polynomial_degree + 1):
            poly_features.append(X ** degree)

        return np.concatenate(poly_features, axis=1)

    def visualize_feature_importance(self, feature_names=None):
        """Plot feature importance using Seaborn."""
        feature_importance = np.array(self._get_coefficients())

        if feature_names is None:
            feature_names = [f'Feature {i + 1}' for i in range(len(feature_importance))]

        expanded_feature_names = self._expand_feature_names(feature_names)
        feature_weight_dict = {name: 0 for name in expanded_feature_names}

        for name, weight in zip(expanded_feature_names, feature_importance):
            feature_weight_dict[name] += abs(weight)

        sorted_features = sorted(feature_weight_dict.items(), key=lambda x: x[1], reverse=True)
        coefs_df = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])

        max_importance = coefs_df["Importance"].max()
        min_importance = coefs_df["Importance"].min()

        if max_importance > min_importance:
            normalized_importance = (coefs_df["Importance"] - min_importance) / (max_importance - min_importance)
            color_indices = (normalized_importance * 255).astype(int)
        else:
            color_indices = np.full(len(coefs_df), 128)

        colors = [plt.cm.Greens(idx / 255) for idx in color_indices]

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=coefs_df["Importance"], y=coefs_df["Feature"], palette=colors)

        for index, value in enumerate(coefs_df["Importance"]):
            label = f"{value:.4f}"
            if abs(value) < 0.05 * max_importance:
                ax.text(value + (0.05 * max_importance), index, label,
                        va='center', ha='left', fontsize=12, color='black', fontweight='bold')
            else:
                ax.text(value * 0.95, index, label,
                        va='center', ha='right' if value > 0 else 'left', fontsize=9, color='white', fontweight='bold')

        plt.xlabel('Feature Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title('Feature Importance Plot', fontsize=16)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.show()

    def _expand_feature_names(self, feature_names):
        """Expand feature names for polynomial features."""
        if self.polynomial_degree is None or self.polynomial_degree == 1:
            return feature_names

        expanded_feature_names = []
        for degree in range(1, self.polynomial_degree + 1):
            expanded_feature_names.extend([f"{col}^{degree}" for col in feature_names])

        return expanded_feature_names

    def _get_coefficients(self):
        """Return model coefficients (excluding bias)."""
        return self.weights[1:]
    
    
class LassoRegularization:
    """Implements L1 (Lasso) regularization."""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * np.sum(np.abs(weights))

    def gradient(self, weights):
        return self.alpha * np.sign(weights)

class RidgeRegularization:
    """Implements L2 (Ridge) regularization."""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * np.sum(np.square(weights))

    def gradient(self, weights):
        return self.alpha * 2 * weights

class LassoRegressor(CustomRegression):
    """Implements Lasso Regression using L1 Regularization."""
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(regularization=LassoRegularization(alpha=alpha), **kwargs)

class RidgeRegressor(CustomRegression):
    """Implements Ridge Regression using L2 Regularization."""
    def __init__(self, alpha=0.1, **kwargs):
        super().__init__(regularization=RidgeRegularization(alpha=alpha), **kwargs)