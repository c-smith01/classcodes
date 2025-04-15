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

        self.data = pd.read_csv(data_root)
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
        self.data = self.data.dropna()
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype('category').cat.codes
                
        print("Columns after preprocessing:", self.data.columns.tolist())

    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test 
        data.
        
        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''
        X_data = data.drop(columns="y").to_numpy()
        y_data = data['y'].astype('category').cat.codes.to_numpy()
        return X_data, y_data


'''
Porblem A-2: Classification Tree Inplementation
'''
class ClassificationTree:
    '''
    You are asked to implement a simple classification tree from scratch. This class will be tested on the
    pre-built enviornment with only numpy and pandas available.

    You may add more variables and functions to this class as you see fit.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''
        def __init__(self, split=None, left=None, right=None, prediction=None):
            '''
            split: tuple - (feature_idx, split_value, is_categorical)
                - For numerical features: split_value is the threshold
                - For categorical features: split_value is a set of categories for the left branch
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

    def __init__(self, random_state: int):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.tree_root = None

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Implement the impurity measure of your choice here. Return the impurity value.
        '''
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-9))  # entropy
        
    def build_tree(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Implement the tree building algorithm here. You can recursivly call this function to build the 
        tree. After building the tree, store the root node in self.tree_root.
        '''
        if len(np.unique(y)) == 1 or depth >= max_depth:
            values, counts = np.unique(y, return_counts=True)
            return self.Node(prediction=values[np.argmax(counts)])

        best_split = self.search_best_split(X, y)
        if best_split is None:
            values, counts = np.unique(y, return_counts=True)
            return self.Node(prediction=values[np.argmax(counts)])

        feature_idx, split_val = best_split
        left_mask = X[:, feature_idx] <= split_val
        right_mask = ~left_mask

        left = self.build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)

        return self.Node(split=(feature_idx, split_val, False), left=left, right=right)

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Implement the search for best split here.

        Expected return:
        - tuple(int, float): Best feature index and split value
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
                        best_split = (feature_idx, t)
        return best_split
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree_root = self.build_tree(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict classes for multiple samples.
        
        Args:
            X: numpy array with the same columns as the training data
            
        Returns:
            np.ndarray: Array of predictions
        '''
        return np.array([self.predict_sample(x, self.tree_root) for x in X])
    
    def predict_sample(self, x: np.ndarray, node: Node):
        if node.is_leaf():
            return node.prediction
        idx, val, _ = node.split
        if x[idx] <= val:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)


def train_XGBoost() -> dict:
    data_loader = DataLoader(data_root="bank-3.csv", random_state=42)
    data_loader.data_prep()
    data_loader.data_split()
    
    X_train, y_train = data_loader.extract_features_and_label(data_loader.data_train)
    X_val, y_val = data_loader.extract_features_and_label(data_loader.data_valid)

    alpha_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    best_f1 = -1
    best_model = None
    best_alpha = None

    for alpha in alpha_vals:
        all_preds = []
        for _ in range(100):  # bootstrapping
            idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_bootstrap = X_train[idx]
            y_bootstrap = y_train[idx]

            model = XGBClassifier(reg_alpha=alpha, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
            model.fit(X_bootstrap, y_bootstrap)

            y_pred = model.predict(X_val)
            f1 = compute_macro_f1(y_val, y_pred)
            all_preds.append(f1)

        avg_f1 = np.mean(all_preds)
        print(f"Alpha={alpha}, Avg F1={avg_f1:.4f}")
        
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model = model
            best_alpha = alpha

    print(f"Best Alpha: {best_alpha}, Best F1: {best_f1:.4f}")
    
    # Save to global model for Gradescope
    global my_best_model
    my_best_model = best_model

    return {"best_alpha": best_alpha, "best_f1": best_f1}


def compute_macro_f1(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1_scores = []

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp + 1e-9)  # avoid division by zero
        recall = tp / (tp + fn + 1e-9)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_scores.append(f1)

    return np.mean(f1_scores)

def plot_roc_and_auc(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    probs = model.predict_proba(X_val)
    classes = np.unique(y_val)
    aucs = []

    plt.figure()
    for i, cls in enumerate(classes):
        y_true_bin = (y_val == cls).astype(int)
        y_score = probs[:, i]

        # Sort by score
        desc_score_indices = np.argsort(-y_score)
        y_true_bin = y_true_bin[desc_score_indices]
        y_score = y_score[desc_score_indices]

        # Compute TPR and FPR
        tpr = []
        fpr = []
        thresholds = np.unique(y_score)
        P = y_true_bin.sum()
        N = len(y_true_bin) - P

        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            tp = ((y_pred == 1) & (y_true_bin == 1)).sum()
            fp = ((y_pred == 1) & (y_true_bin == 0)).sum()
            tpr.append(tp / (P + 1e-9))
            fpr.append(fp / (N + 1e-9))

        tpr = np.array(tpr)
        fpr = np.array(fpr)
        auc = np.trapz(tpr, fpr)
        aucs.append(auc)

        plt.plot(fpr, tpr, label=f"Class {cls} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - One vs. Rest")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")  # Saves instead of displaying
    print("ROC curve saved as 'roc_curve.png'.")

    return np.mean(aucs)




'''
Initialize the following variable with the best model you have found. This model will be used in testing 
in our pre-built environment.
'''
my_best_model = XGBClassifier()


if __name__ == "__main__":
    results = train_XGBoost()
    print(results)
    
    # Define alpha values as stated in Part B.1
    alpha_vals  = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    
     # Initialize data
    data_loader = DataLoader(data_root="bank-3.csv", random_state=42)
    data_loader.data_prep()
    data_loader.data_split()

    # Extract train and validation features/labels
    X_train, y_train = data_loader.extract_features_and_label(data_loader.data_train)
    X_val, y_val = data_loader.extract_features_and_label(data_loader.data_valid)

    # Train classification tree
    clf_tree = ClassificationTree(random_state=42)
    clf_tree.fit(X_train, y_train)

    # Predict on validation data
    y_pred = clf_tree.predict(X_val)

    # Evaluate performance
    f1 = compute_macro_f1(y_val, y_pred)
    print(f"Macro F1 Score on validation set: {f1:.4f}")
    
    results = train_XGBoost()
    auc = plot_roc_and_auc(my_best_model, X_val, y_val)
    print(f"Macro-average AUC: {auc:.4f}")