# Pandas Set Options
pd.set_options('display.max_rows', 100)
pd.set_option("display.max_columns", None) # Displays all columns in pandas dataframe
pd.set_option("display.precision", 3)

# Reading DataFrame from CSV
df = pd.read_csv(<FILENAME>)

# Summary Stats
df.head()
df.tail()
df.shape
df.describe()
df.describe(include='all').T
df.<CAT_VAR>.unique()
df[<CAT_VAR>].value_counts() # counts how many values this cat var has.
corr = df.corr() # correlation

# Checking Values
df.isna().sum().sort_values(ascending=False)

# Categorial Columns / Values
categorical_columns = list(df.dtypes[df.dtypes == "O"].index.values)

# ReCast non-numerical/object columns as CAT_VARS.
for column in categorical_columns:
    df[column] = df[column].astype("category")

# Dropping Columns
df.drop(columns=['name', 'data'], inplace=True)

# Dropping Nans
df.dropna()

# ReOrder Columns
cols = list(df.columns)
cols.remove(<THIS_COL>)
cols.inser(<THIS_COL>, 0)
df = df.reindex(columns=cols

# Rename Columns
new_cols = {"old_1": "new_1", "old_2", "new_2",}
df.rename(colums=new_cols, inplace=True)

# ReMap Cat Vars
df[<CATVAR>] = df[<CAT_VAR>].map({"catvar1": "CV1", "catvar2": "CV2"...})

# Visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Histogram
df.loc[:, ['x1', 'x2', 'x3'].hist(bins=20, figsize=(10,10))
df[data_columns].hist(figsize=(16,4), layout=(1,6))

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

## Pairplotting
sns.pairplot(df, hue="class") # for pairwise comparisons

# Regression Models

## Standardization
from Sklearn.preprocessing import StandardScaler
df_std[data_columns] = StandardScaler().fit_transform(df[data_columns])


## PCA
from sklearn.decomposition import PCA
df_pca = PCA().fit(df_std[data_columns]) 

