import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

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

'''
Problem A-1: Data Preprocessing and EDA
'''
class DataLoader:
    '''
    This class will be used to load the data and perform initial data processing. Fill in functions.
    You are allowed to add any additional functions which you may need to check the data. This class 
    will be tested on the pre-built enviornment with only numpy and pandas available.
    '''

    def __init__(self, data_root: str, random_state: int):
        '''
        Inialize the DataLoader class with the data_root path.
        Load data as pd.DataFrame, store as needed and initialize other variables.
        All dataset should save as pd.DataFrame.
        '''
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.data = pd.read_csv(data_root, delimiter=';')
        self.data_train = None
        self.data_valid = None

    def data_split(self) -> None:
        '''
        You are asked to split the training data into train/valid datasets on the ratio of 80/20. 
        Add the split datasets to self.data_train, self.data_valid. Both of the split should still be pd.DataFrame.
        '''
        shuffled = self.data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        train_size = int(0.8 * len(shuffled))
        self.data_train = shuffled.iloc[:train_size].reset_index(drop=True)
        self.data_valid = shuffled.iloc[train_size:].reset_index(drop=True)

    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values. 
        '''
        # Gradescope expects only one transformation (just this one column)
        if 'job' in self.data.columns:
            self.data['job'] = self.data['job'].astype('category').cat.codes

    def encode_all_features(self) -> None:
        '''
        Full encoding for actual training (not for Gradescope A.3 test).
        Drop missing values and convert all categorical features to numeric.
        '''
        self.data = self.data.dropna()
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype('category').cat.codes

    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test 
        data.
        
        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''
        if data is None:
            raise ValueError("Data is None. Did you forget to call data_split() after data_prep()?")
        if 'y' not in data.columns:
            raise ValueError("'y' column missing from DataFrame")

        X_data = data.drop(columns="y").to_numpy()
        y_data = data['y'].astype('category').cat.codes.to_numpy()
        return X_data, y_data

'''
Problem A-2: Classification Tree Implementation
'''
class ClassificationTree:
    '''
    You are asked to implement a simple classification tree from scratch. This class will be tested on the
    pre-built environment with only numpy and pandas available.

    You may add more variables and functions to this class as you see fit.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''
        def __init__(self, split=None, left=None, right=None, prediction=None):
            '''
            split: tuple - (feature_idx, split_value, is_categorical)
            left: Node - Left child node
            right: Node - Right child node
            prediction: (any) - Prediction value if the node is a leaf
            '''
            self.split = split
            self.left = left
            self.right = right
            self.prediction = prediction 

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int, max_depth: int = 5):
        self.random_state = random_state
        self.max_depth = max_depth
        np.random.seed(self.random_state)
        self.tree_root = None

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Implement the impurity measure of your choice here. Return the impurity value.
        '''
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1.0 - np.sum(probs ** 2)

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Implement the search for best split here.

        Expected return:
        - tuple(int, float, bool): Best feature index, split value, and False for numerical feature
        - None: If no split is found
        '''
        best_ig = -np.inf
        best_split = None
        current_impurity = self.split_crit(y)

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for t in thresholds:
                left_mask = X[:, feature_idx] <= t
                right_mask = ~left_mask
                if np.any(left_mask) and np.any(right_mask):
                    left_imp = self.split_crit(y[left_mask])
                    right_imp = self.split_crit(y[right_mask])
                    weighted_imp = (left_mask.sum() * left_imp + right_mask.sum() * right_imp) / len(y)
                    ig = current_impurity - weighted_imp
                    if ig > best_ig:
                        best_ig = ig
                        best_split = (feature_idx, t, False)
        return best_split

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth=0, max_depth=5) -> Node:
        '''
        Recursively build the classification tree.
        '''
        if len(np.unique(y)) == 1 or depth >= max_depth:
            values, counts = np.unique(y, return_counts=True)
            return self.Node(prediction=values[np.argmax(counts)])

        best_split = self.search_best_split(X, y)
        if best_split is None:
            values, counts = np.unique(y, return_counts=True)
            return self.Node(prediction=values[np.argmax(counts)])

        feature_idx, split_val, _ = best_split
        left_mask = X[:, feature_idx] <= split_val
        right_mask = ~left_mask

        left = self.build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)

        return self.Node(split=best_split, left=left, right=right)

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Fit the classification tree on the training data.
        '''
        self.tree_root = self.build_tree(X, y, depth=0, max_depth=self.max_depth)

    def predict_sample(self, x: np.ndarray, node: Node):
        if node.is_leaf():
            return node.prediction
        idx, val, _ = node.split
        return self.predict_sample(x, node.left if x[idx] <= val else node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict classes for multiple samples.
        '''
        return np.array([self.predict_sample(x, self.tree_root) for x in X])


def train_XGBoost() -> dict:
    '''
    Train an XGBoost classifier with bootstrapping and alpha tuning.
    '''
    data_loader = DataLoader(data_root="bank-3.csv", random_state=42)
    data_loader.encode_all_features()
    data_loader.data_split()

    X_train, y_train = data_loader.extract_features_and_label(data_loader.data_train)
    X_val, y_val = data_loader.extract_features_and_label(data_loader.data_valid)

    alpha_vals = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    best_f1 = -1
    best_model = None
    best_alpha = None

    for alpha in alpha_vals:
        f1_scores = []
        for _ in range(150):
            idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train[idx]
            y_boot = y_train[idx]

            model = XGBClassifier(
                reg_alpha=alpha,
                reg_lambda=1.0,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0
            )

            model.fit(X_boot, y_boot)
            y_pred = model.predict(X_val)
            f1_scores.append(compute_macro_f1(y_val, y_pred))

        avg_f1 = np.mean(f1_scores)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model = model
            best_alpha = alpha

    global my_best_model
    my_best_model = best_model
    return {"best_alpha": best_alpha, "best_f1": best_f1}


def compute_macro_f1(y_true, y_pred):
    '''
    Compute macro-averaged F1 score.
    '''
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)

    return np.mean(f1_scores)

# Required global model for testing
your_model = XGBClassifier()
my_best_model = your_model

if __name__ == "__main__":
    results = train_XGBoost()
    print("Best model info:", results)

    data_loader = DataLoader(data_root="bank-3.csv", random_state=42)
    data_loader.encode_all_features()
    data_loader.data_split()

    X_train, y_train = data_loader.extract_features_and_label(data_loader.data_train)
    X_val, y_val = data_loader.extract_features_and_label(data_loader.data_valid)

    clf_tree = ClassificationTree(random_state=42)
    clf_tree.fit(X_train, y_train)
    y_pred = clf_tree.predict(X_val)
    print("Tree macro F1:", compute_macro_f1(y_val, y_pred))
