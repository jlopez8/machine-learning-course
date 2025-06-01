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

# Adding a row:
df.loc[<ROW_NAME>] = <ARRAY_LIKE>

# Adding a column:
df[<COL_NAME>] = <ARRAY_LIKE>

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

# One-Hot Encoding / Get Dummies for Cat Vars
pd.get_dummies(df[<CAT_VAR>], prefix=<CAT_VAR>)

# Merging
# Left Merge, left index means we use df1s index and right index means we use df2s index.
# These are keywords specifying on what exactly we will be joining. In this case, on the index.
df1.merge(df2, left_index=True, right_index=True)