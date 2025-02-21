from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import traceback as tb
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE

def load_data(file_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset split into features and targets"""
    try:
        df = pd.read_csv(file_path)
        y = df[target_column]
        X = df.drop(columns=[target_column])
    except Exception as e:
        print("An error occurred while loading the data:")
        print(tb.format_exc())
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float=0.2, 
               random_state: int=50) -> tuple:
    """Split data into training and testing sets."""
    try:
         X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size=test_size, random_state=random_state)
    except:
        print("An error occurred:")
        print(tb.format_exc())
    return X_train, X_test, y_train, y_test
        
    
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, 
                random_state: int=50, k_neighbors: int=1) -> tuple:
    """Apply SMOTE to balance the classes."""
    try:
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    except:
        print("An error occurred:")
        print(tb.format_exc())
    return smote.fit_resample(X_train, y_train)

def plot_accuracy(models: pd.DataFrame):
    """Plots the accuracy scores of different models"""
    try:
        plt.figure(figsize=(12, 8))
        models.sort_values(by="Accuracy", ascending=True)["Accuracy"].plot(kind="barh")
        plt.title("Classification Model Accuracy")
        plt.xlabel("Accuracy")
        plt.ylabel("Model")
        plt.subplots_adjust(left=.2)
        plt.show()
    except Exception as e:
        print("An error occurred while plotting the accuracy scores:")
        print(tb.format_exc())

    
def main():
    #load dataframe and define X and y
    X, y = load_data(file_path="disease_symptom_original_cleaned_smote_2.csv", target_column="prognosis")
    
    #split data for training and testing
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=.2)
    
    #balencing classes using SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    #create and fit models
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=50)
    models, predictions = clf.fit(X_train_balanced, X_test, y_train_balanced, y_test)
    
    #print model summary
    print("Model Performance Summary:")
    print(models)
    
    #finding the best model
    best_model_name = models.sort_values('Accuracy', ascending=False).iloc[0].name
    print(f"\nBest Model: {best_model_name}")
    
    #plotting accuracy scores
    plot_accuracy(models)

if __name__ == "__main__":
    main()