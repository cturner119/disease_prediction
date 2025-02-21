from sklearn.preprocessing import LabelEncoder
import pandas as pd
import traceback as tb
from imblearn.over_sampling import SMOTE
from ydata_profiling import ProfileReport

def drop_rare_labels(df: pd.DataFrame, target_column: str, min_count: int = 2) -> pd.DataFrame:
    """Drop rows where the target column has fewer than `min_count` occurrences."""
    try:
        label_counts = df[target_column].value_counts()
        labels_to_keep = label_counts[label_counts >= min_count].index
    except:
        print("An error occurred:")
        print(tb.format_exc())
    return df[df[target_column].isin(labels_to_keep)]

def display_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Display summary information about the DataFrame with an optional name.

    Args:
        df (pd.DataFrame): The DataFrame to display information for.
        name (str): The name of the DataFrame (default is "DataFrame").
    """
    try:
        print(f"\n=== Information for {name} ===")
        print(f"\nDataFrame shape:\n {df.shape}")
        null_values = df.isnull().sum()     
        print(f"\nNull value count:\n {null_values[null_values > 0]}")
        duplicated_values = df.duplicated().sum()                        
        print(f"\nDuplicate value count:\n {duplicated_values}")
        print("\nDataFrame Info:")
        print(df.info())         
    except:
        print("An error occurred:")
        print(tb.format_exc())

def create_profile_report(df: pd.DataFrame, title: str):
    """Generate a profile report."""
    try:
        report = ProfileReport(df, title= title)
        report.to_file(title + ".html")
    except:
        print("An error occurred:")
        print(tb.format_exc())
    
def main():
    #loading data
    df1 = pd.read_csv("Training.csv")
    df2 = pd.read_csv("Testing.csv")

    #combining test and train DataFrames
    df = pd.concat([df1, df2])
    
    #printing DataFrame information
    display_dataframe_info(df,name="Original Concatendated DataFrame")

    #drop duplicates
    df = df.drop_duplicates()
    
    #print the count of each class 
    print(f"\nOriginal label counts:\n{df["prognosis"].value_counts()}")
    
    #dropping labels with less than 5 counts
    df = drop_rare_labels(df, target_column='prognosis', min_count=5)

    #label encoding
    label_encoder = LabelEncoder()
    df["prognosis"] = label_encoder.fit_transform(df["prognosis"])

    #resetting the index
    df.reset_index(drop=True, inplace=True)
    
    #define X and y for SMOTE
    y = df["prognosis"]
    X = df.drop(columns=["prognosis"])
    
    #apply SMOTE 
    smote = SMOTE(random_state=50, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    #create SMOTE encoded Dataframe
    df_smote = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                          pd.Series(y_resampled, name="prognosis")], axis=1)
    
    #drop SMOTE duplicates
    df_smote = df_smote.drop_duplicates()
    
    #resetting the index
    df_smote.reset_index(drop=True, inplace=True)
    
    #checking Dataframe info
    display_dataframe_info(df_smote, name="Encoded DataFrame with SMOTE")
    
    #print new y label counts
    print(f"\nSMOTE label counts:\n{df_smote["prognosis"].value_counts()}")
    
    #create new CSV for encoded labels and SMOTE
    # df_smote.to_csv("disease_symptom_original_cleaned_smote_2.csv", index=False)
    
    #decode encoded labels
    y_decoded = label_encoder.inverse_transform(y_resampled)
    
    #create new Dataframe for SMOTE with labels
    df_smote_labels = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                                 pd.Series(y_decoded, name="prognosis")], axis=1)
    
    #dropping duplicates
    df_smote_labels = df_smote_labels.drop_duplicates()
    
    #resetting index
    df_smote_labels.reset_index(drop=True, inplace=True)
    
    #check Dataframe info 
    display_dataframe_info(df_smote_labels, name="DataFrame with SMOTE and Labels")
    
    #print new y label counts
    print(f"\nSMOTE label counts:\n{df_smote_labels['prognosis'].value_counts()}")
    
    #create new dataset with smote and labels
    df_smote_labels.to_csv("disease_symptom_original_cleaned_smote_labels_2.csv", index=False)
    
    #creating profile report for original dataset
    create_profile_report(df= df_smote_labels, title="Predict Disease Symptom Report")
    
if __name__ == "__main__":
    main()
