import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 2D plotting
fig = plt.figure(figsize=(8, 8/1.618))

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
fig.show()

