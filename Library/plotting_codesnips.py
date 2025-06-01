import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from yellowbrick.regressor import residuals_plot
from yellowbrick.regressor import prediction_error
from yellowbrick.regressor import AlphaSelection
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay


# Figures and Subplots
fig = plt.figure(figsize=(8, 8/1.618))
fig, ax = plt.subplots(figsize=(8, 8 / 1.618))

# 2D plotting
sns.scatterplot(x=<C1>, y=<C2>, hue=<CLASS>, data=df)

plt.scatter(
    df[<COL1>], df[<COL2>],
    s=100, c=df[<CLASS>].apply(lambda x: color_map[x])
)

# 3D plotting
fig = plt.figure(figsize=(8, 8/1.618));
ax = fig.add_subplot(111, projection="3d");

color_map = {<DICT_OF_COLORMAP>}

# matplot
plt.scatter(
    df[<COL1>], df[<COL2>], zs=df[<COL3>], 
    s=100, c=df[<CLASS>].apply(lambda x: color_map[x])
)

# plotly express
fig = px.scatter_3d(df, x=<C1>, y=<C2>, z=<C3>, color=<CLASS>);
fig.update_traces(marker_size=<MARKER_SIZE>)
fig.show()

# Image show
# NOTE: for showing mages. 
# X = samples with raw data / features. each row is an image with 64 "bits" indicating color of an 8x8 image.
# y = targets / interpretations of each sample.
plt.imshow(X[0].reshape(8, 8), cmap="Greys")
print(f"y interpreted value: {y[0]}")

# Visualization: Histograms, Boxplots, Cat + Violin, HeatMap, etc.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Histogram
df.loc[:, ['x1', 'x2', 'x3'].hist(bins=20, figsize=(10,10))
df[data_columns].hist(figsize=(16,4), layout=(1,6))

### Histogram w Seaborn.
### Verification of ML fits for a linear regression.
fig, ax = plt.subplots(figsize=(8, 8 / 1.618));
ax.set_xlim([-15, 15])
sns.distplot(result.resid, bins=30)
plt.show()

fig, ax = plt.subplots(figsize=(8, 8 / 1.618));
sns.histplot(result.resid, bins=30);
ax.set_xlim([-15, 15])
plt.show()

### Histograms w Yellowbrick
### Verification of ML fits for a linear regression.
plt.figure(figsize=(8, 8 / 1.618))
visualizer = residuals_plot(linear_regression, X_train, y_train, X_test, y_test, is_fitted=True)
plt.show()

### w Seaborn
sns.histplot(df, x=data_columns[0], hue="class", bins=20, ax=axs[0,0],);
sns.histplot(df, x=data_columns[1], hue="class", bins=20, ax=axs[0,1],);
sns.histplot(df, x=data_columns[2], hue="class", bins=20, ax=axs[1,0],);
sns.histplot(df, x=data_columns[3], hue="class", bins=20, ax=axs[1,1],);

## Boxplot
df.boxplot()

### as subplots in same figure
fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns
df.boxplot(ax=axs[0,0], column=data_columns[0], by="class", figsize=(5,3))
df.boxplot(ax=axs[0,1], column=data_columns[1], by="class", figsize=(5,3));
df.boxplot(ax=axs[1,0], column=data_columns[2], by="class", figsize=(5,3),);
df.boxplot(ax=axs[1,1], column=data_columns[3], by="class", figsize=(5,3),);
plt.show()

## Cat and Violin plot
sns.catplot(data=df, kind='bar', x='x1', y='y1', hue='x2')
sns.violinplot(data=df, x='x1', y='y1', hue='reponse var/x2', split=True)

## Heatmap
mask = np.triu(np.ones_like(corr, d_type=bool) # mask is for missing values
plt.subplots(figsize=(10,8))
sns.heatmap(corr, mask=mask, cmap='seismic', center=0, square='True')
# Visualize weights w heatmap to check strength of weights.
fig = plt.figure(figsize=(20, 2));
sns.heatmap(lda_weights.abs(), annot=True, cmap="Blues", annot_kws={"size": 12})

## Pairplotting
sns.pairplot(df, hue="class") # for pairwise comparisons

## Verification of ML fits for a linear regression.
## Uses yellowbrick.
### QQ Plots
visualizer = residuals_plot(linear_regression, X_train, y_train, X_test, y_test, is_fitted=True, qqplot=True, hist=False)
plt.show()

### y_hat vs y_test 
visualizer = prediction_error(linear_regression, X_test, y_test, is_fitted=True)

### Ridge Regression Lasso Regression with Cross Validation
visualization = AlphaSelection(RidgeCV(alphas=lambdas))
visualization = AlphaSelection(LassoCV(alphas=lambdas))
visualization.fit(X_train, y_train)

# Confusion Matrix Display
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(lda_model, X_test, y_test, display_labels=["M", "B"])

# ROC Curve
from sklearn.metrics import RocCurveDisplay
fig = plt.figure(figsize=(8, 8 / 1.618));
RocCurveDisplay.from_estimator(lda_model, X_test, y_test, ax=plt.subplot())