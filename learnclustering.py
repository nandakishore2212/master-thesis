import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform

# generate two clusters: a with 100 points, b with 50:
#np.random.seed(4711)  # for repeatability of this tutorial
#a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
#b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
#X = np.concatenate((a, b),)
data = pd.read_csv('C:\\Users\\nandu\\PycharmProjects\\MasterArbeit\\Lvpdtw.csv', header = None)
print(data.shape)
X1 = data
X1 = X1.iloc[1:]
#X1.drop(X1.columns[[0]], axis=1, inplace=True)
X1 = X1.drop(X1.columns[0], axis =1)
print (X1.shape)  # 150 samples with 2 dimensions
#plt.scatter(X[:,0], X[:,1])

X = squareform(X1)
print(X.shape)
# generate the linkage matrix
Z = linkage(X, 'average')

c, coph_dists = cophenet(Z, X)
print(c)

# calculate full dendrogram
#plt.figure(figsize=(15, 8))
plt.figure()
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i,d,c in zip(ddata['icoord'],ddata['dcoord'],ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()
