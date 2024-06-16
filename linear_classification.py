import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    """
    Prepares the dataset for training by removing specified columns,
    converting boolean columns to numeric, and one-hot encoding categorical variables.
    Splits the data into training and testing sets.

    Args:
    df (DataFrame): The input dataframe.
    train_size (float): Proportion of the dataset to include in the train split.
    shuffle (bool): Whether to shuffle the data before splitting.
    random_state (int): Controls the shuffling applied to the data before the split.

    Returns:
    X_train (DataFrame): Training features.
    X_test (DataFrame): Testing features.
    y_train (Series): Training target.
    y_test (Series): Testing target.
    """
    # Drop specified columns
    X = df.drop(columns=["Month", "Browser", "OperatingSystems"])

    # Convert boolean columns to numeric (0 and 1)
    bool_columns = X.select_dtypes(include=bool).columns
    X[bool_columns] = X[bool_columns].astype(int)

    # One-hot encode categorical variables
    categorical_columns = ["Region", "TrafficType", "VisitorType"]
    X = pd.get_dummies(X, columns=categorical_columns)

    # Separate features and target variable
    target = 'Revenue'
    X = X.drop(columns=[target])
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

def linear_transformation(df):
    """
    Prepares the data by calling prepare_data and scales the features using MinMaxScaler.

    Args:
    df (DataFrame): The input dataframe.

    Returns:
    X_train_scaled (array): Scaled training features.
    X_test_scaled (array): Scaled testing features.
    y_train (Series): Training target.
    y_test (Series): Testing target.
    """
    # Call prepare_data with a 70%-30% train-test split and seed=42
    X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.7, random_state=42)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Fit the scaler on the training data and transform both train and test sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def main():
    # Load the dataset
    df = pd.read_csv("project2_dataset.csv")

    # Prepare and scale the data
    X_train, X_test, y_train, y_test = linear_transformation(df)

    # Initialize the custom logistic regression model
    model = CustomLogisticRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Model evaluation
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Print accuracy scores
    print("Ευστοχία στο σύνολο εκπαίδευσης:", train_accuracy)
    print("Ευστοχία στο σύνολο δοκιμής:", test_accuracy)

    # Compute and print the confusion matrix
    confusion_mat = confusion_matrix(y_test, y_test_pred)
    print("Πίνακας Σύγχυσης:")
    print(confusion_mat)

class CustomLogisticRegression(LogisticRegression):
    def __init__(self, *args, **kwargs):
        """
        Custom logistic regression model that sets the maximum number of iterations to 1000
        and removes the penalty argument if provided.
        """
        # Remove penalty argument if provided
        kwargs.pop('penalty', None)
        # Set the maximum number of iterations to 1000
        kwargs['max_iter'] = 1000
        super().__init__(*args, **kwargs)

if __name__ == "__main__":
    main()
