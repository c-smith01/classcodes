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
    def __init__(self, train_dat_root: str, test_dat_root: str):
        """Initialize data processor with paths to train and test data.
        
        Args:
            data_root: root path to data directory
        """
        self.train_dat_root = train_dat_root
        self.test_dat_root = test_dat_root
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data from CSV files.
        
        Returns:
            Tuple containing training and test dataframes
        """
        # TODO: Implement data loading
        train_dataframe = pd.read_csv(self.train_dat_root)
        test_dataframe = pd.read_csv(self.test_dat_root)
            
        return train_dataframe, test_dataframe
            
        
    def check_missing_values(self, data: pd.DataFrame) -> int:
        """Count number of missing values in dataset.
        
        Args:
            data: Input dataframe
        
        Returns:
            Number of missing values
        """
        
        # TODO: Implement missing value check
        return data.pd.isnull().sum().sum()
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values.
        
        Args:
            data: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # TODO: Implement data cleaning
        return data.dropna()
        
    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataframe, convert to numpy arrays.
        
        Args:
            data: Input dataframe
            
        Returns:
            Tuple of feature matrix X and label vector y
        """
        # TODO: Implement feature/label extraction
        X = data.drop(columns=['PT08.S1(CO)']).values
        y = data['PT08.S1(CO)'].values
        return X, y
    
class LinearRegression:
    def __init__(self, learning_rate=1, max_iter=10):
        """Initialize linear regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train linear regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training
        n_samps, n_feats = X.shape
        self.weights = np.zeros(n_feats)
        self.bias = 0
        losses_list = []
        
        i=1
        while i<self.max_iter:
            y_estim = np.dot(X, self.weights) + self.bias
            loss = np.mean((y-y_estim)^2) + self.bias
            losses_list.append(loss)
            
            weight_gradient = (-2/n_samps) * np.dot(X.T, (y-y_estim))
            bias_gradient = (-2/n_samps) * np.sum(y-y_estim)
            
            self.weights -= self.learning_rate * weight_gradient
            self.bias    -= self.learning_rate * bias_gradient
            i+=1
        
        return losses_list
            
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement linear regression prediction
        return X@self.weights + self.bias #np.dot(X, self.weights) + self.bias

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MSE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
        return np.mean((y_true-y_pred)**2)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation
        return np.sqrt(self.criterion(y_true, y_pred))

class LogisticRegression:
    def __init__(self, learning_rate=1, max_iter=10):
        """Initialize logistic regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train logistic regression model with normalization and L2 regularization.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        losses_list = []
        
        i=1
        while i<self.max_iter:
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            losses_list.append(loss)
            
            grad_w = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            grad_b = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            i+=1
            
        return losses_list
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction probabilities using normalized features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction
        return self.label_binarize(self.predict_proba(X))

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate BCE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with handling of edge cases.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """
        # TODO: Implement F1 score calculation
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * (precision * recall) / (precision + recall)

    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize labels for binary classification.
        
        Args:
            y: Target vector
            
        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization
        return (y >= 0.5).astype(int)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate AUROC.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation
        #return np.trapz(tpr, fpr)

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
        def cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
            scores = []
            for train_idx, val_idx in self.kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(model.metric(y_val, y_pred))
            return scores

if __name__ == "__main__":
    #print("Hello World!")
    
    ### 1 Data Processing ###
    prcsr = DataProcessor("data_train_25s.csv","data_test_25s.csv")
    train_dat, test_dat = prcsr.load_data()
    train_dat = prcsr.clean_data(train_dat)
    X_train, y_train = prcsr.extract_features_labels(train_dat)
    
    ### 2 Exploratory Data Analysis ###
    # Histograms of all data
    train_dat.hist(figsize=(12, 10), bins=30, edgecolor='black')
    plt.tight_layout()
    plt.show()
    
    # Two features for comparison
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=train_dat["NOx(GT)"], y=train_dat["NO2(GT)"])
    plt.xlabel("NOx (GT)")
    plt.ylabel("NO2 (GT)")
    plt.title("Correlation between NOx and NO2 Levels")
    plt.show()
    
    # Pearson's Correlation???
    
    lin_regress = LinearRegression()
    lin_regress.fit(X_train, y_train)
    
    log_regess = LogisticRegression()
    
    evltr = ModelEvaluator()
    print(evltr.cross_validation(lin_regress, X_train, y_train))
    print(evltr.cross_validation(log_regess, X_train, y_train))
    
    ### 6 ROC Curve - Logistic Regression
    folds = [1,2,3,4,5]
    ROCs = []
    #mlp.plot
    #plt.show()
    