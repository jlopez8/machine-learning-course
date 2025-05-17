import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.api as sm
from scipy import stats


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

# Training

## 1. Linear Regression
# Get the model. 
linear_regression = LinearRegression()

# Train the model with our training data.
# This sets the weights for the model which is stored in linear_regression object.
linear_regression.fit(X_train, y_train)

# Check some early prediction results based on this fit.
y_verify_pred = pd.DataFrame(linear_regression.predict(X_train))

(y_verify_pred - y_train)[0]

# Accessing weights
weights = linear_regression.coef_
intercept = linear_regression.intercept_

## 2. Ordinary Least Squares via  statsmodels package.
glm = sm.OLS(y_train, sm.add_constant(X_train))

result = glm.fit() # Fit model to the training data provided to the object.
result.summary() # Show summary statistics of trained model.

# Verification

# These fits can be verified visually by the histogram techniques as outlined in the 
# plotting_codesnips.py file in section: ### Verification of ML fits for a linear regression.

