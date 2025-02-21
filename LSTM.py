import pandas as pd
import traceback as tb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import random
from tensorflow.keras.optimizers import Adam


np.random.seed(50)
random.seed(50)
tf.random.set_seed(50)

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

def evaluate_model(model: Sequential, X_test_reshaped: pd.DataFrame,
                   y_test: pd.Series) -> tuple:
    """Evaluate the model and return metrics."""
    try:
        y_pred_probs = model.predict(X_test_reshaped)
        y_pred = np.argmax(y_pred_probs, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
    except:
        print("An error occurred:")
        print(tb.format_exc())
    return accuracy, report, cm

def plot_results(accuracy: float, report: str, cm: pd.DataFrame):
    """Plot evaluation results including accuracy, classification report, and confusion matrix."""
    try:
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print(f"confusion matrix:\n{cm}")
    except:
        print("An error occurred:")
        print(tb.format_exc())

def plot_history(history):
    """Plot training and validation loss and accuracy."""
    try:
        plt.figure(figsize=(12, 5))

        #loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("RNN Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()

        #accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("RNN Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except:
        print("An error occurred:")
        print(tb.format_exc())
    
def plot_confusion_matrix(cm: pd.DataFrame, xlabel: str, ylabel: str,
                          title: str, xtick: tuple, ytick: tuple):
    """Plot the confusion matrix."""
    try:
        plt.figure(figsize=(12, 8))
        sns.set_context("notebook", font_scale=.5)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=(xtick), 
                    yticklabels=(ytick),
                    cbar=True,
                    linewidths=0.5, 
                    linecolor='black',
                    square=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.show()
    except:
        print("An error occurred:")
        print(tb.format_exc())

def main():
    #load dataset and define target
    X, y = load_data(file_path="disease_symptom_original_cleaned_smote_2.csv", 
              target_column="prognosis")
    
    #splitting data to test and training sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    #balencing classes using SMOTE
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    #reshape data for RNN
    X_train_balanced = X_train_balanced.values.reshape((X_train_balanced.shape[0], 1, X_train_balanced.shape[1]))
    X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    #defining parameters
    rnn_units = 32
    activation_func = "softmax"
    loss_function = "sparse_categorical_crossentropy"
    num_features = X_train_balanced.shape[2]
    num_classes = len(np.unique(y_train_balanced))
    dropout_rate = 0.3
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    learning_rate = 0.003
    optimizer = Adam(learning_rate=learning_rate)
    
    #build the model
    model = Sequential()

    #RNN layer
    model.add(LSTM(rnn_units, input_shape=(1, num_features), return_sequences=False, 
                   kernel_regularizer=l2(0.02)))

    #add dropout rate
    model.add(Dropout(dropout_rate))
    
    #output layer
    model.add(Dense(num_classes, activation=activation_func))

    #compile the model
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    #model summary
    model.summary()

    #training the model
    history = model.fit(X_train_balanced, y_train_balanced, epochs=50, batch_size=32, validation_split=0.2,
              callbacks = [early_stopping])
    
    #calculating metrics
    accuracy, report, cm = evaluate_model(model, X_test_reshaped, y_test)
    
    #print metrics
    plot_results(accuracy, report, cm)
    
    #plotting history
    plot_history(history)
    
    #plot confusion matrix
    class_labels = sorted(y.unique())
    plot_confusion_matrix(cm, xlabel="Prognosis Predicted Labels", ylabel="Prognosis True Labels",
                            title="RNN Disease Prediction Confusion Matrix",
                            xtick=(class_labels),
                            ytick=(class_labels))

if __name__ == "__main__":
    main()