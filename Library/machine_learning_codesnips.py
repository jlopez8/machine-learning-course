import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame(<FILENAME>)

# Methods
def remove_missing_and_nonNumerical_values(X, y):
    """
    Removes missing and non-numerical values from dataframe 'X' and corresponding reponse variable y. 
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

# Separate into features and response variables.
## Take everything except the response var.
X = df.loc[:, df.columns != <RESPONSE_COL>]

## Now assign response variable only.
y = df.loc[:, df.columns == <RESPONSE_COL>]

## Train/Test Data Split
# 67% / 33% SPLIT
# Random state is for seeding a partition. 0 - 42 are common seeds.
X_train, X_test, y_train, y_text = train_test_split(X, y, test_size= 0.33, random_state=42)


