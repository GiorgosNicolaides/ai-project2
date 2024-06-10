import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

def prepare_data(df, train_size=None, shuffle=True, random_state=None):
    # Remove specified features
    X = df.drop(columns=["Month", "Browser", "OperatingSystems"])

    # Convert boolean columns to numeric
    bool_columns = X.select_dtypes(include=bool).columns
    X[bool_columns] = X[bool_columns].astype(int)

    # One-hot encode categorical variables
    categorical_columns = ["Region", "TrafficType", "VisitorType"]
    X = pd.get_dummies(X, columns=categorical_columns)

    target = 'Revenue'
    X = X.drop(columns=[target])
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=shuffle,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test

def linear_transformation(df):
    # Call prepare_data with 70%-30% split and seed=42
    X_train, X_test, y_train, y_test = prepare_data(df, train_size=0.7, random_state=42)

    # MinMaxScaler
    scaler = MinMaxScaler()
    # Fitting the scaler on the training data and transforming both train and test sets
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def undersampling():
    df = pd.read_csv("project2_dataset.csv")
    X_train, X_test, y_train, y_test = linear_transformation(df)

    # Apply undersampling to the training set
    undersampler = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

    model = CustomLogisticRegression()
    model.fit(X_train_resampled, y_train_resampled)

    y_train_pred = model.predict(X_train_resampled)
    y_test_pred = model.predict(X_test)

    # Model evaluation
    train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print("Ευστοχία στο σύνολο εκπαίδευσης:", train_accuracy)
    print("Ευστοχία στο σύνολο δοκιμής:", test_accuracy)

    confusion_mat = confusion_matrix(y_test, y_test_pred)
    print("Πίνακας Σύγχυσης:")
    print(confusion_mat)



class CustomLogisticRegression(LogisticRegression):
    def __init__(self, *args, **kwargs):
        # remove penalty
        kwargs.pop('penalty', None)
        # set iteration number to 1000
        kwargs['max_iter'] = 1000
        super().__init__(*args, **kwargs)

undersampling()