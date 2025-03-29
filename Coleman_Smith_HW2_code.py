import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

'''
Problem: University Admission Classification using SVMs

Instructions:
1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
   the library specified in question instruction available. Importing additional libraries will result in 
   compilation errors and you will lose marks.

2. Fill in the skeleton code precisely as provided. You may define additional 
   default arguments or helper functions if necessary, but ensure the input/output format matches.
'''
class DataLoader:
    '''
    Put your call to class methods in the __init__ method. Autograder will call your __init__ method only. 
    '''
    
    def __init__(self, data_path: str):
        """
        Initialize data processor with paths to train dataset. You need to have train and validation sets processed.
        
        Args:
            data_path: absolute path to your data file
        """
        
        # TODO：complete your dataloader here!
        df = pd.read_csv(data_path)
        df = self.create_binary_label(df)
        
        feature_list = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA']
        target = 'label'
        
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target]) # 80/20 dataset split
        
        scaler = StandardScaler() # use StandardScaler from sklearn to normalize data
        train_set[feature_list] = scaler.fit_transform(train_set[feature_list])
        test_set[feature_list] = scaler.transform(test_set[feature_list])
        
        self.train_data = train_set
        self.test_data = test_set
    
    def create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create a binary label for the training data.
        '''
        
        median_chance = df['Chance of Admit'].median()
        df['label'] = (df['Chance of Admit']>median_chance).astype(int)
        
        return df

class SVMTrainer:
    def __init__(self):
        self.models = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray, kernel: str, **kwargs) -> SVC:
        '''
        Train the SVM model with the given kernel and parameters.

        Parameters:
            X_train: Training features
            y_train: Training labels
            kernel: Kernel type
            **kwargs: Additional arguments you may use
        Returns:
            SVC: Trained sklearn.svm.SVC model
        '''
        
        model = SVC(kernel=kernel, **kwargs)
        model.fit(X_train,y_train)
    
        return model
    
    def get_support_vectors(self,model: SVC) -> np.ndarray:
        '''
        Get the support vectors from the trained SVM model.
        '''
        model.support_vectors_
        
    # def accuracy_score(self,y_test,y_pred):
    #     '''
    #     Determine correct # of admit predictions for given model
    #     '''
    #     correct_preds = y_pred.intersection(y_test)
    #     accuracy = correct_preds.shape[0]/y_pred.shape[0]
    #     print(accuracy) # for debugging
    #     return accuracy
    
'''
Initialize my_best_model with the best model you found.
'''

my_best_model = SVC(kernel='rbf', C=10, gamma=0.1)

if __name__ == "__main__":
    print("Hello, World!")
    
    datldr = DataLoader(data_path='data.csv')
    trnr = SVMTrainer()
    
    train_data = datldr.train_data
    test_data = datldr.test_data
    
    feature_sets = [
    ['CGPA', 'SOP'],
    ['CGPA', 'GRE Score'],
    ['SOP', 'LOR'],
    ['LOR', 'GRE Score']
    ]
    
    kernel_sets = {
    'linear': {},
    'rbf': {'C': 1, 'gamma': 'scale'},
    'poly': {'degree': 3, 'C': 1}
    }
    
    best_model = None
    best_accuracy = 0
    best_config = ('', [])
    
    for feats in feature_sets:
        X_train = datldr.train_data[feats].values
        y_train = datldr.train_data['label'].values
        X_test = datldr.test_data[feats].values
        y_test = datldr.test_data['label'].values
        
        for kernel_type, params in kernel_sets.items():
            model = trnr.train(X_train,y_train,kernel=kernel_type,**params)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            svs = trnr.get_support_vectors(model)
            
            print(f'{kernel_type.upper()} on {feats} - Accuracy = {accuracy:.3f}')
            print(f'Support vectors: {svs}')
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_config = (kernel_type, feats)
    
    print(f"\n Best model: {best_config[0].upper()} kernel on {best_config[1]} with accuracy {best_accuracy:.3f}")            