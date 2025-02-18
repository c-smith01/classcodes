import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple, List

'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional 
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow 
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''

class DataProcessor:
    def __init__(self, data_root: str):
        """Initialize data processor with paths to train and test data.
        
        Args:
            data_root: root path to data directory
        """
        self.data_root = data_root
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from CSV files.
        
        Returns:
            Tuple containing training and test dataframes
        """
        # TODO: Implement data loading
        
    def check_missing_values(self, data: pd.DataFrame) -> int:
        """Count number of missing values in dataset.
        
        Args:
            data: Input dataframe
            
        Returns:
            Number of missing values
        """
        # TODO: Implement missing value check
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values.
        
        Args:
            data: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # TODO: Implement data cleaning
        
    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe, convert to numpy arrays.
        
        Args:
            data: Input dataframe
            
        Returns:
            Tuple of feature matrix X and label vector y
        """
        # TODO: Implement feature/label extraction
    
class LinearRegression:
    def __init__(self):
        """Initialize linear regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """
        self.weights = None
        self.bias = None
        self.learning_rate = None
        self.max_iter = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train linear regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement linear regression prediction

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation

class LogisticRegression:
    def __init__(self):
        """Initialize logistic regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
        """
        self.weights = None
        self.bias = None
        self.learning_rate = None
        self.max_iter = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate BCE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
    
    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with handling of edge cases.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """
        # TODO: Implement F1 score calculation

    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize labels for binary classification.
        
        Args:
            y: Target vector
            
        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation

class ModelEvaluator:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize evaluator with number of CV splits.
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
    def cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        """Perform cross-validation
        
        Args:
            model: Model to be evaluated
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of metric scores
        """
        # TODO: Implement cross-validation

if __name__ == "__main__":
    print("Hello World!")