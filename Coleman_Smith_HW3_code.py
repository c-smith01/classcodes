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
        df = self.data
        train_set, test_set = df.random_split([0.8, 0.2])   # 80/20 dataset split
        return train_set, test_set

    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values. 
        '''
        data = self.data
        data = data.dropna()
        return data

    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test 
        data.
        
        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''
        X_data = data.features(1:4)
        y_data = data.features(4)
        
        return X_data,y_data


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
        pass
        
    def build_tree(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Implement the tree building algorithm here. You can recursivly call this function to build the 
        tree. After building the tree, store the root node in self.tree_root.
        '''
        pass

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Implement the search for best split here.

        Expected return:
        - tuple(int, float): Best feature index and split value
        - None: If no split is found
        '''
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict classes for multiple samples.
        
        Args:
            X: numpy array with the same columns as the training data
            
        Returns:
            np.ndarray: Array of predictions
        '''
        pass


def train_XGBoost() -> dict:
    '''
    See instruction for implementation details. This function will be tested on the pre-built enviornment
    with numpy, pandas, xgboost available.
    '''
    pass

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



'''
Initialize the following variable with the best model you have found. This model will be used in testing 
in our pre-built environment.
'''
my_best_model = XGBClassifier()


if __name__ == "__main__":
    print("Hello World!")
    
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