import pandas as pd
import traceback as tb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import optuna
from optuna.exceptions import TrialPruned
from plotly.io import show

def load_data(file_path: str, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset split into features and targets"""
    try:
        df = pd.read_csv(file_path)
        y = df[target_column]
        X = df.drop(columns=[target_column])
    except:
        print("An error occurred:")
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
  
def main():
    #load dataset and defining target and features
    X, y = load_data(file_path="disease_symptom_original_cleaned_smote_2.csv", 
                     target_column="prognosis")
        
    #splitting data to test and training sets
    X_train, X_test, y_train, y_test = split_data(X, y)
        
    #balencing classes using SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    target_accuracy = 1.0
        
    def objective(trial):
        """This function runs trials to find optimum hyperparameters for
        the Random Forest classifier and returns the accuracy"""
        n_estimators = (trial.suggest_int("n_estimators", 100, 1000))
        max_depth = (trial.suggest_int("max_depth", 10, 150))
        min_samples_split = (trial.suggest_int("min_samples_split", 2, 32))
        min_samples_leaf = (trial.suggest_int("min_samples_leaf", 1, 32))
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            
        model = RandomForestClassifier(n_estimators = n_estimators, 
                                    max_depth = max_depth,
                                    min_samples_split = min_samples_split,
                                    min_samples_leaf = min_samples_leaf,
                                    max_features= max_features,
                                    random_state=42)
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        if accuracy >= target_accuracy:
            raise TrialPruned(f"Reached target accuracy: {accuracy}")
    
        return accuracy
        
    #defining trails, and create study
    n_trials = 20        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
        
    #printing best parameters and accuracy score
    best_params = study.best_params
    best_score = study.best_value
    print("Best Parameters:", best_params)
    print("Best Accuracy Score:", round(best_score, 4))

    #plotting parameter visualization
    fig1 = optuna.visualization.plot_optimization_history(study)
    show(fig1)
    fig2 = optuna.visualization.plot_parallel_coordinate(study)
    show(fig2)
    fig3 = optuna.visualization.plot_slice(study, params=["n_estimators", "max_depth", 
                                                       "min_samples_split", "min_samples_leaf"] )
    show(fig3)
    fig4 = optuna.visualization.plot_param_importances(study)
    show(fig4)

if __name__ == "__main__":
    main()
    
    