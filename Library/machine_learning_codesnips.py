import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

from scipy import stats
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.api as sm

from Library import data # This is a custom library.

######### 
# Data: Retrieve
#########
df = pd.DataFrame(<FILENAME>)
df = data.get(<FILENAME>)

#########
# Data: Train, Test, Split
#########

# Separate into features and response variables.
## Take everything except the response var.
X = df.loc[:, df.columns != <RESPONSE_COL>]

## Now assign response variable only.
y = df.loc[:, df.columns == <RESPONSE_COL>]

## Train/Test Data Split
# 67% / 33% SPLIT
# Random state is for seeding a partition. 0 - 42 are common seeds.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, random_state=42)

# Train Test split with stratification.
# Suppose you have data that fits into 4 categories.
# With stratification, you can take this into account so the data splitting works evenly across
# those 4 categories, otherwise, it will split in a random manner and you may have 
# much of the training data significantly concentrated on samples that are not represented in the
# testing data etc. So your models are not going to perform well on the test since they were trained
# in the other data.
text_train, text_test, labels_train, labels_test = train_test_split(text, labels, test_size=0.30, random_state=42, stratify=labels)

#########
# Data: Remove missing, nonnumerical, and scale data.
#########
# NOTE: This could be in the ML library more explicitly, but it is being handled in the
# data custom library.

# Remove missing and categorical/nonnumerical values
X_train, y_train = data.remove_missing_and_nonNumerical_values(X_train, y_train)
X_test, y_test = data.remove_missing_and_nonNumerical_values(X_test, y_test)

# Scale
X_train, y_trian, scaler = data.scale_data(X_train, y_train)
X_test, y_test, _ = data.scale_data(X_test, y_test, scaler=scaler)


##########
# Model Training
##########

#NOTE: it is generally good to have scaled data for this phase of the processes.

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

######################
# K Fold Cross Validation
######################

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


## As Applied using Lasso Regression

# set-up the lambdas. 
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 10, 50, 100]
lasso_cross_validation_metrics = pd.DataFrame(columns=["mean_mse", "mean_norm_mse", "mean_r2"])

# Cross validation for each lambda
for lambda_ in lambdas:

    # Set-up the cross validation metric storage arrays. These are used per fold.
    cv_mse = []
    cv_norm_mse = []
    cv_r2 = []

    kfold = KFold(n_splits=5)
    # i = 1 # This is for printing statistics per fold.
    for train_index, test_index in kfold.split(X_train):

        # Get train and test data for this fold.
        x_train_fold = X_train.values[train_index]
        y_train_fold = y_train.values[train_index]
        x_test_fold = X_train.values[test_index]
        y_test_fold = y_train.values[test_index]

        # Fit the model to this fold.
        lasso_model = Lasso(alpha=lambda_)
        lasso_model.fit(x_train_fold, y_train_fold)

        # Get the prediction data.
        y_hat_fold = lasso_model.predict(x_test_fold)

        # Compute performance metrics.
        mse_fold = mean_squared_error(y_test_fold, y_hat_fold)
        r2_fold = r2_score(y_test_fold, y_hat_fold)
        norm_mse_fold = 1 - r2_fold

        # Store performance metrics.
        cv_mse.append(mse_fold)
        cv_norm_mse.append(norm_mse_fold)
        cv_r2.append(r2_fold)

    # Compute and store mean of performance metrics.
    lasso_cross_validation_metrics.loc[f"Lambda: {lambda_}"] = [np.mean(cv_mse), np.mean(cv_norm_mse), np.mean(cv_r2)]

# Display and sort by best fit
lasso_cross_validation_metrics.sort_values(by="mean_r2", ascending=False)

## As Applied using Lasso Regression Scikit-learn pre-built library
lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 10, 50, 100]
lasso_cv = LassoCV(alphas=lambdas)
lasso_cv.fit(X_train, y_train)

# NOTE: this is slightly different for lasso versus ridge. 
# You have to compute this. It is not done automatically like in Ridge.
# This essentially performs the same process AGAIN, but focusing on the score.
kfold = KFold(n_splits=5)
lasso_r2 = np.mean(cross_val_score(lasso_cv, X_train, y_train, cv=kfold))
print(f"Best Lambda: {lasso_cv.alpha_: 3.3f} R2_score: {lasso_r2: 3.3f}")

# Alpha Selection for Lasso Regression Cross Validation
# See plotting codesnips for more using Alpha Selection.

######################
# One-Hot Encoding
######################

categorical_columns = list(df.dtypes[df.dtypes == "category"].index.values)
for column in categorical_columns:
    df_one_hot = pd.get_dummies(df[column], prefix=column, dtype="int")
    df = df.merge(df_one_hot, left_index=True, right_index=True)
df.drop(columns=categorical_columns, inplace=True)


######################
# Classifiers
######################

# Linear Discriminant Analysis LDS
lda = LinearDiscriminantAnalysis()

# Now fit the model.
lda_model= lda.fit(X_train, y_train)

# Analyze the weights.
lda_weights_ = lda_model.coef_
lda_weights = pd.DataFrame(lda_weights_, columns=X.columns)

# NAIVE Bayes Classifier
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes_cv_score = cross_val_score(naive_bayes, X_train, y_train, cv=10)
naive_bayes_mean_cv = np.mean(naive_bayes_cv_score)
naive_bayes_model = GaussianNB().fit(X_train, y_train)
naive_bayes_theta = naive_bayes_model.theta_
means = pd.DataFrame(naive_bayes_theta, columns=<ORIGINAL_DATA_COLUMNS>)

#Quadratic Classifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

# Model.
qda_model = QuadraticDiscriminantAnalysis()
# Cross Validation
qda_model_cv_score = cross_val_score(qda_model, X_train, y_train, cv=10)
qda_model_mean_cv = np.mean(qda_model_cv_score)
# Optimize for regularization.
param = {"reg_param": np.linspace(0, 1, 21, endpoint=True)}
# Apply Grid search, n_jobs is number of processors, -1 being all of them. 
# This produces a model.
qda_grid_search = GridSearchCV(qda_model, param, cv=10, n_jobs=-1, refit=True)
qda_grid_search.fit(X_train, y_train)

#Logistic Classifier
# For more information see lesson 15. 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Model. Default paramters is L2 penatly with weight of 1.
logistic_regression = LogisticRegression(max_iter=1000)
# Cross validation
logistic_regression_cv_score = cross_val_score(logistic_regression, X_train, y_train, cv=10)
logistic_regression_mean_cv = np.mean(logistic_regression_cv_score)
# Optimize for Regularization with Grid Scan.
# There are two types of penalties, Lasso and Ridge, so we can specify both or either in this case.
# C is the inverse of regularization strength, and must be a positive float.
# This will prime the logistic_regression with paramaters to try when fitting.
# NOTE: default solver "lbfgs" no support L1. We can specify support using "liblinear" to support both L1 and L2 
# in the params with "solver" as shown here.
search_space = 10 ** np.linspace(-3, 3, 21, endpoint=True)
param = {"penalty": ["L1", "L2"], "C": search_space, "solver": ["liblinear"]}
logistic_regression_grid_search = GridSearchCV(logistic_regression, param, cv=10, n_jobs=-1, refit=True)

# Fit.
logistic_regression_grid_search.fit(X_train, y_train)

# Accessing weights
weights = logistic_regression_grid_search.best_estimator_.coef_ 

######################
# Time Series
######################

# Time Series Windows
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

w = 4

# Windows
## Remember this should lead to a matrix of windows.
## Each of this data should therefore expand both the X-features so they are in the 
## manner of sliding window and the corresponding y-targets.
## This part here will instantiate the windows_train instance of `sliding_window_view`.
## This this givs us the corresponding w time-features used to predict the w+1 position for EACH of the
## Sliding window partitions. 
windows_train = sliding_window_view(y_train_scaled, w + 1, axis=0).copy()
X_train_w = windows_train.squeeze()[:,:-1] # Take all up to the last one.
y_train_w = windows_train.squeeze()[:,-1] # Take the last one.
windows_test = sliding_window_view(y_test_scaled, w + 1, axis=0).copy()
X_test_w = windows_test.squeeze()[:, :-1] 
y_test_w = windows_test.squeeze()[:, -1]

######################
# SVM Kernels
######################

## Linear Kernel
from sklearn.svm import SVR
from skopt import BayesSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
# Set your hyperparameters for the SVM linear kernel. 
param = {"C": 10 ** np.linspace(-3, 3, 101), "epsilon": np.linspace(0, 0.1, 11)}
# SVR cache_size is MB of size of the kernel cache.
linear_support_vector_regression = SVR(kernel="linear", max_iter=25000, cache_size=2000)
# Optimize our Hyperparameters using an optimization method.
# In BayesSearchCV we have a cross-validation splitting strategy that uses a time series split.
# This is different that a static, non-temporal split. We need to carefully define this strategy
# so it lines up with our current strategy. The sliding window we have is size w + 1, so this should be the gap. 
# Scoring is also a function provided to this method. We need to wrap our mean squared error in a make_scorer
# wrapper to give to the scoring method of Bayes Search. Since this method is such that lower is better, we can 
# provide argument greater is better to be False.
# Refit: after performing the hyperparameter search, the model will have the best parameters 
linear_support_vector_search = BayesSearchCV(
    linear_support_vector_regression, param, n_iter=15, 
    cv=TimeSeriesSplit(n_splits=5, gap=w+1), 
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    n_jobs=-1,
    refit=True,
    random_state=0
)
# Finally, fit the train data. 
linear_support_vector_fit = linear_support_vector_search.fit(X_train_w, y_train_w)

## Polynomial Kernel
params = {
    "C": 10 ** np.linspace(-3, 3, 101), 
    "epsilon": np.linspace(0, 0.1,  11),
    "degree": [2, 3, 4],
}
polynomial_support_vector_regression = SVR(
    kernel="poly", max_iter=25000, cache_size=2000
)
polynomial_svm_search = BayesSearchCV(
    polynomial_support_vector_regression,
    params, cv=TimeSeriesSplit(n_splits=5, gap=w+1),
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    n_jobs=-1, refit=True, random_state=0
)
polynomial_svm_fit = polynomial_svm_search.fit(X_train_w, y_train_w)

## Radial Basis Function Kernel
params = {
    "C": 10 ** np.linspace(-3, 3, 101),
    "epsilon": np.linspace(0, 0.1, 11), 
    "gamma": ["scale", "auto"]
}
radial_basis_function_regression = SVR(
    kernel="rbf", max_iter=25000, cache_size=2000
)
radial_basis_function_search = BayesSearchCV(
    radial_basis_function_regression, 
    search_spaces=params,
    n_iter=15, 
    cv=TimeSeriesSplit(n_splits=5, gap=w+1),
    scoring=make_scorer(mean_squared_error, greater_is_better=False),
    n_jobs=-1, refit=True, random_state=0
)
radial_basis_function_fit = radial_basis_function_search.fit(X_train_w, y_train_w)

# Performance Metrics
linear_support_vector_mse = mean_squared_error(y_test_w, linear_support_vector_fit.predict(X_test_w))
linear_support_vector_mae = mean_absolute_error(y_test_w, linear_support_vector_fit.predict(X_test_w))
print(f"linear_support_vector_mse: {linear_support_vector_mse}.")
print(f"linear_support_vector_mae: {linear_support_vector_mae}.")
# Visualize:
## See plotting codesnip.

######################
# Metrics
######################

# Classification Report
from sklearn.metrics import classification_report
ConfusionMatrixDisplay.from_estimator(lda_model, X_test, y_test, display_labels=["M", "B"])

# Confusion Matrix (Display)
## See: plotting codesnips.

# ROC Curve (Display)
## See: plotting codesnips.

