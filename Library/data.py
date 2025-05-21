# data.py
# Class for handling data in preprocessing, storing, and retrieving. 

import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler

def get_data(filename, rename_columns={}, remap={}, combinator={}, **kwargs):
    df = pd.read_csv(filename, index_col=kwargs.get("index_col", None))
    df.rename(columns=rename_columns, inplace=True)

    # Remap values using cat map.
    for column, col_map in remap.items():
        df[column] = df[column].map(col_map)

    # Creates a custom sum-wise combination of specific columns.
    for combo, columns in combinator.items():
        df[combo] = df[columns].sum(axis=1)
    return df

def remove_missing_and_nonNumerical_values(X, y):
    """
    Removes missing and non-numerical values from dataframe 'X' and CORRESPONDING rows from reponse variable y. 
    Prints minor statistics such as original shapes, categorical variables removed, and new sizes.

    Parameters
    ------
    X (pd.DataFrame): Original dataframe. Numerical and non-numerical datatypes must be pre-assigned as 
    numerical type and category.
    y (pd.DataFrame): Original response variable.

    Returns
    ------
    X (pd.DataFrame): Reduced dataframe with non nans or categorical column variables.
    y (pd.DataFrame): Reduced response variable matching corresponding rows of reduced dataframe 'X'.
    """
    print(f"Original Size X: {X.shape} y: {y.shape}")

    # Drop Categorical Values. Nonnumerical Dtypes must be 
    categorical_columns = X.dtypes[X.dtypes=="category"].index.values
    X = X.drop(columns=categorical_columns)
    print("Removed {}".format(categorical_columns))

    # Drop Missing Values.
    X = X.dropna()
    # Make sure you reduce y as well, since the above is a row-reduction technique.
    y = y[X.index]
    print(f"New Size X: {X.shape} y: {y.shape}")
    return X, y

def scale_data(X, y, scaler=None):
    """
    Scales data according to specified scaler. If None, will start and use a new scaler: MinMaxScaler.
    Missing and non-numerical values are dropped during this process. 

    Parameters
    -------
    X (pd.DataFrame): Dataframe of features. 
    y (pd.DataFrame): Dataframe of respones.
    scaler (scaler): Optional. Scaler from scikitlearn.preprocessing._data library. Default is MinMaxScaler.

    Returns
    -------
    X (pd.DataFrame): Dataframe of rescaled features. 
    y (pd.DataFrame): Dataframe of rescaled responses.
    scaler (scaler): Scaler used in the scaling process.
    """
    if not scaler:
        scaler = MinMaxScaler()
    X, y = remove_missing_and_nonNumerical_values(X, y)
    X[X.columns] = scaler.fit_transform(X[X.columns])
    return X, y, scaler

def save_weights(weights, filename) -> None:
    """
    Save weights from a model regression. 

    Parameters
    ------
    weights (array-like): Array of weights.
    filename (str): Path to filename for saving weights. 

    Returns
    ------
    (None)
    """
    with open(filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        for w in weights:
            csv_writer.writerow([w])
    print(f"Weights saved here:\n{filename}")
    return 

def load_weights(filename) -> list:
    """
    Save weights from a model regression. 

    Parameters
    ------
    filename (str): Path to filename for loading weights. 

    Returns
    ------
    weights (list): List of loaded weights.
    """
    weights = []
    with open(filename, "r", newline="") as file:
        csv_reader  = csv.reader(file)
        for x in csv_reader:
            weights.append(float(x[0]))
    return weights

