import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import stats


df = pd.DataFrame(<FILENAME>)

# Custom Methods
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

glm_fitted = glm.fit() # Fit model to the training data provided to the object.
glm_fitted.summary() # Show summary statistics of trained model.

# Verification

# These fits can be verified visually by the histogram techniques as outlined in the 
# plotting_codesnips.py file in section: ### Verification of ML fits for a linear regression.

# Metrics for Performance

## MSE
N = X_train.shape[0] # Get from training data rows.
y_hat = glm_fitted.predict(sm.add_constant(X_train)) # Use OLS fit from stats package.
mse_stat = ((y_hat - y_train) ** 2).sum()/ N 
mse_sk = mean_squared_error(y_train, y_hat) # Option: use scikit-learn.

## Normlized MSE. Uses the variance subpackage of numpy.
sigma = np.var(y_train)
norm_mse = mse_stat / sigma

## R-Squared
R_squared = 1 - norm_mse
R_squared_sk = r2_score(y_train, y_hat) # Option: use scikit-learn


# K Fold Cross Validation

## As applied with an OLS model
cross_validation_metrics = pd.DataFrame(columns=["MSE", "norm_MSE", "R2"])

# Instantiate a KFold object with 5 splits.
# This really only contains information about INDICES of the folds, not the DATA itself, you have to 
# call in the data using the index information provided by the KFold split. These will all come
# from the X_train data set, NOT THE TEST DATASET!! Rememember, that is not used until the final
# verification. 
kfold = KFold(n_splits=5)

# Perform the k-fold fitting on each of the folds. This retains the MSE statistics per fold.
i = 1
for train_index, test_index in kfold.split(X_train):
    print(f"Split: {i}\n\tTest folds: {i}\n\tTrain folds: {[j for j in range(1, kfold.n_splits + 1) if j != i]}")

    # Get the kfold data.
    x_train_fold = X_train.values[train_index]
    y_train_fold = y_train.values[train_index]
    x_test_fold = X_train.values[test_index]
    y_test_fold = y_train.values[test_index]

    # Perform model regression with KFold data.
    linear_regression = LinearRegression().fit(x_train_fold, y_train_fold)
    y_hat_fold = linear_regression.predict(x_test_fold)

    # Calculate Metrics of Performance
    mse_fold = mean_squared_error(y_test_fold, y_hat_fold)
    normalized_mse_fold = 1 - r2_score(y_test_fold, y_hat_fold)
    r2_fold = r2_score(y_test_fold, y_hat_fold)
    print(f"\tMSE: {mse_fold: 3.3f} normalized_MSE: {normalized_mse_fold: 3.3f} R2: {r2_fold: 3.3f}")

    # Store Cross Validation Metrics
    cross_validation_metrics.loc[f"Fold {i}", :] = [mse_fold, normalized_mse_fold, r2_fold]
    i+=1

# Adds mean of each column in the metrics dataframe tracker.
cross_validation_metrics.loc["Mean", :] = cross_validation_metrics.mean()


## As Applied with Ridge Regression

# Track the ridge cross validation
ridge_cross_validation_metrics = pd.DataFrame(columns=["mean MSE", "mean norm MSE", "mean R2"])
lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 10, 50, 100]

# Calculate Cross validation for each lambda.
for lambda_ in lambdas:
    k_fold = KFOLD(n_splits=5)

    cv_mse = []
    cv_norm_mse = []
    cv_r2 = []

    # Calculat the metric for each partition and take the mean.
    i = 1
    for train_index, test_index in k_fold.split(X_train):
        # option to print the test fold data
        # print(f"Lambda: {lambda)}: Split: {i}\n\tTest folds: {i}\n\tTrain folds: {[j for j in range(1, k_fold.n_splits + 1) if j != 1]}")

        x_train_fold = X_train.values[train_index]
        y_train_fold = y_train.values[train_index]
        x_test_fold = X_train.values[test_index]
        y_test_fold = y_train.values[test_index]

        # Perform the model regression w kfold data.
        ridge_reg = Ridge(alpha=lambda_).fit(x_train_fold, y_train_fold)
        y_hat_fold = ridge_reg.predict(x_test_fold)

        # Calculate the metrics of performance.
        mse_fold = mean_squared_error(y_test_fold, y_hat_fold)
        r2_fold = r2_score(y_test_fold, y_hat_fold)
        norm_mse_fold = 1 - r2_fold
        print(f"MSE fold: {mse_fold: 3.3f} Norm MSE fold: {norm_mse_fold: 3.3f} R2 fold: {r2_fold: 3.3f}")

        # Capture the values of cross validation at the specific fold.
        # These are used later to calculate a mean.
        cv_mse.append(mse_fold)
        cv_norm_mse.append(norm_mse_fold)
        cv_r2.append(r2_fold)

    # Calculate the mean of the cross validation metrics at this lambda. Store in the row of the validation metrics df.
    ridge_cross_validation_metrics.loc[f"lambda: {lambda_}"] = [np.mean(cv_mse), np.mean(cv_norm_mse), np.mean(cv_r2)]

# Now display and sort by best R2 fit.
ridge_cross_validation_metrics.sort(by="mean R2", ascending=False)

## As Applied using Ridge Regression Scikit-learn pre-built library
lambdas = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1, 10, 50, 100]
# NOTE: fit directly on the training data because it will perform the folding for us.
ridge_cv = RidgeCV(alphas=lambdas, cv=5).fit(X_train, y_train)

# Alpha Selection for Ridge Regression Cross Validation
# See plotting codesnips for more using Alpha Selection.

