from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd


# Get Data
df = pd.csv(<FILENAME>)


# Preprocessing data

# Formatting:
df.rename(columns={<NEWCOLUMNS_DICTIONARY>}, inplace=True)
df[<COL1>] = df[<COL1>].map({<NEWVALUES_MAP>})
data_columns = [<COLUMNS_WITH_DATA_OF_INTEREST>]

df[<NEW_CONCAT_COL_FOR_CATEGORIES_CLASSIFICATION>] = 
df[<CATVAR1>] + df[<CATVAR2>]


# Standardizing/Scaling Data:
## Good for distance-based dim redux.
df_scaled = df.copy(deep=True)
df_scaled[data_columns] = MinMaxScaler((0,1)).fit_transform(df_scaled[data_columns])
df_std[data_columns] = StandardScaler().fit_transform(df_scaled[data_columns])

# Perform Dim Redux:
df_pca = PCA().fit(df_std[data_columns]) 
df_pca.transform(df_std[data_columns]);
df_std[["PC1", "PC2", "PC3"]] = df_pca.transform(df_std[data_columns]);

tsne = TSNE(n_components=3, perplexity=10, n_iter=2000, init="random")
df_scaled[["TSNE1", "TSNE2", "TSNE3"]] = tsne.fit_transform(df_scaled[data_columns])

lle = LocallyLinearEmbedding(n_components=3, n_neighbors=15)
df_scaled[["LLE1", "LLE2", "LLE3"]] = lle.fit_transform(df_scaled[data_columns])
print("Reconstruction Error\n", lle.reconstruction_error_)

mds = MDS(n_components=3, metric=True, n_init=15)
df_scaled[["MDS1", "MDS2", "MDS3"]] = mds.fit_transform(df_scaled[data_columns])
print("MDS 3D Stress aka MSE:\n", mds.stress_)

isomap = Isomap(n_components=3, n_neighbors=10)
df_scaled[["ISOMAP1", "ISOMAP2", "ISOMAP3"]] = isomap().fit_transform(df_scaled[data_columns])

# Note: this requires a standardized dataset.
fisher = LinearDiscriminantAnalysis(n_components=2)
df_std[["FISH1", "FISH2"]] = fisher.fit_transform(df_std[data_columns].values, y=df_std[<TARGET_REGRESSION_COL>])


# Dim Redux for Support Vector Machines
from sklearn.decomposition import PCA

# Will use PCA to dim redux the text data, then we will hotglue it with the labels to make a dataframe.
# Make sure you make the vectorized matrix dense for the fit_transform.
# Note the use of np.asarray, which this requires because fit transform does no like np.matrix.
pca = PCA()
text_data_as_dense_array = np.asarray(text_train_vectorized.todense())
text_data = pca.fit_transform(text_data_as_dense_array)
df_text = pd.DataFrame(text_data[:, :2])
df_labels = pd.DataFrame({"labels": labels_train})
df_train = pd.concat([df_text, df_labels], axis=1)
# Visualization see plotting codesnips. 