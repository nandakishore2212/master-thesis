import pandas as pd
import peakutils
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import time
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree, cut_tree
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
#sys.setrecursionlimit(10000)
np.set_printoptions(threshold=sys.maxsize)

############# Getting data from CSV file ####################
#Extracting data - Each variable is a column in the file
#start= time.time()
bayerdata = pd.read_csv('D:\\acads\\semester4\\thesis\\BAYER data\\T521-48-R.csv', sep = ' ', header = None)
#timearray = bayerdata[0:3600000][0]
bp = bayerdata[0:3600000][2]

numbers = list(range(0,3600000))

#Converting to numpy array for easier processing
#timearray = np.array(timearray)
bp = bpcopy= np.array(bp)


#Savitzky-Golay filter is applied so that mini-peaks due to fluctuations are smoothed out
bp = savgol_filter(bp, 101, 3)




##################### Finding peaks  ##############

##################################### BP #########################################
in_bp =[]
for i in range(7200):
    #Absolute thresholding is not done for BP as there are massive variations.
    if i != 0:
        bp_sm = bp[500*i-1:500*(i+1)+1]
        in_bp_sm = peakutils.indexes(bp_sm, thres=0.6, min_dist=250)
        in_bp_sm_adj = [element + i * 500 - 1 for element in in_bp_sm]
    else:
        bp_sm = bp[500 * i :500 * (i + 1) + 1]
        in_bp_sm = peakutils.indexes(bp_sm, thres=0.6, min_dist=250)
        in_bp_sm_adj = [element + i*500 for element in in_bp_sm]

    if in_bp and in_bp_sm_adj:
        #print(in_bp)
        if (in_bp_sm_adj[0] - in_bp[-1] < 250):
            if (bp[in_bp_sm_adj[0]] > bp[in_bp[-1]]):
                in_bp.pop()
            elif (bp[in_bp[-1]] > bp[in_bp_sm_adj[0]]):
                in_bp_sm_adj = in_bp_sm_adj[1:]
            else:
                #print("values", bp[in_bp_sm_adj[0]], bp[in_bp[-1]])
                in_bp_sm_adj[0] = int((in_bp_sm_adj[0] + in_bp[-1]) / 2)
                in_bp.pop()

    in_bp = in_bp + in_bp_sm_adj
#in_bp = in_bp.tolist()
print(len(in_bp))


rrint_bp = []
for i in range(len(in_bp)-1): #Make an array of RR interval lengths
    rrint_bp.append(in_bp[i+1]-in_bp[i])






clus2 = []
clus10 = []
def hycluster(X, link, metr, datatype):

    # generate the linkage matrix
    Z = linkage(X, link)

    cuttree = cut_tree(Z, n_clusters= [2, 10])
    #print('cut tree shape', cuttree.shape)
    #print('Full cuttree', cuttree)
    global clus2, clus10
    for i in cuttree:
        clus2.append(i[0])
        clus10.append(i[1])

    c, coph_dists = cophenet(Z, X)
    print('Cophenet:',metr,c)
    titl = 'Hierarchical Clustering ' + datatype +','+ link +','+ metr
    # calculate full dendrogram
    #plt.figure(figsize=(15, 8))
    plt.figure()
    def fancy_dendrogram(*args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)
        ptitle = kwargs.pop('plttitle',0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title(ptitle)
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
        #truncate_mode='lastp',
        #p=12,
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,  # useful in small plots so annotations don't overlap
        plttitle = titl
    )

################################bp ######################################################
############################ Compressing each RR-interval to the median length #########################

bpmedian = int(np.median(rrint_bp)) #returns the median length of all the intervals
#print(bpmedian)
#print('bp beat number', len(in_bp))
bpmodfull = []
bpmodmat = np.zeros((len(in_bp)-1,bpmedian))
#print('Mat shape', bpmodmat.shape)
bpmodfull = np.array(bpmodfull)
for i in range(len(in_bp)-1):
    nn = in_bp[i+1]-in_bp[i] #length of a beat
    bpcomp1 = np.linspace(0, nn-1, num=nn, endpoint=True) #Defining an x axis with as many points as the beat length
    bpres1 = bp[in_bp[i]:in_bp[i+1]] #The actual data
    f1 = interp1d(bpcomp1, bpres1, kind='nearest') #A function is defined that interpolates the x axis to the nearest
                                                    #point in the dataset. Right now the two arrays are 1 to 1 (equal size)

    bpxnew = np.linspace(0, nn-1, num= int(bpmedian), endpoint=True) #Now we declare a new X-axis with the same length
                #but the number of points in the axis is equal to the median length. If the median length is lower than
                #axis, it'll be sparsely populated and if higher, more densely.

    bpmod = f1(bpxnew) #Using the same functioned on the new axis finds corresponding points in new axis. Meaning the
            #beat is effectively compressed or stretched.
    #print('Types', type(bpmod), type(bpmodfull))
    bpmodfull = np.append(bpmodfull,bpmod)  #Full array with all beats of same median length
    bpmodmat[i] = bpmod
    #print(len(bpmod))
    #plt.plot(bpcomp1, bpres1, '-', bpxnew, bpmod, 'o')
    #plt.show()

    #print('Len:', len(bpcomp1), len(bpres1))
    #break
    #rrint_bp.append(in_bp[i+1]-in_bp[i])
'''for i in bppeakanomal:
    bpmodmat[bpmedian*i]= bppeaklistavg[bpmedian*i]
    np.delete(bpmodmat, [bpmedian*i:(bpmedian*(i+1))])
'''
#print(bpmodmat.shape)
XvecEuc = pdist(bpmodmat, metric = 'euclidean')
'''
XvecMink = pdist(bpmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(bpmodmat, metric = 'cityblock')
XvecSeuc = pdist(bpmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(bpmodmat, metric = 'correlation')
#XvecHamm = pdist(bpmodmat, metric = 'hamming')
#XvecMah = pdist(bpmodmat, metric = 'mahalanobis', VI=None)
'''


hycluster(XvecEuc,'average','euclidean','bp')

'''
hycluster(XvecMink,'average','Minkowski','bp')
hycluster(XvecSeuc,'average','Standardised Euclidean','bp')
hycluster(XvecCity,'average','City Block','bp')
hycluster(XvecCorr,'average','Correlation','bp')
#hycluster(XvecHamm,'average','Hamming','bp')
#hycluster(XvecMah,'average','Mahalanobis','bp')
'''
print('Peak list length', len(rrint_bp))
anomlist2 =  []
for i in range(len(rrint_bp)):
    if clus2[i] == 1:
        anomlist2.append(i)
print('Number of morph anomalies', len(anomlist2))

#unique2, counts2 = np.unique(clus2, return_counts=True)
#unique, counts = np.unique(clus10, return_counts=True)
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123
def colour_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts



####################################################################################################
start= time.time()
tsnebp = TSNE(n_components=2).fit_transform(bpmodmat)
print(tsnebp.shape)
#print(tsnebp)
plt.figure('TSNE Scatter plot')
print(tsnebp[:,0], tsnebp[:,1])
plt.scatter(tsnebp[:,0], tsnebp[:,1])
plt.title('TSNE Scatter plot')
plt.xlabel('x')
plt.ylabel('y')

clus2 = np.asarray(clus2)
clus10 = np.asarray(clus10)
print('Shapes', tsnebp.shape, clus2.shape)
colour_scatter(tsnebp, clus2)
colour_scatter(tsnebp, clus10)

pca = PCA(n_components=2)
pcabp = pca.fit_transform(bpmodmat)
print ('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
print(pcabp.shape)
#print(pcabp)
plt.figure('PCA Scatter plot')
#print(pcabp[:,0], pcabp[:,1])
plt.scatter(pcabp[:,0], pcabp[:,1])
plt.title('PCA Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
colour_scatter(pcabp,clus2)
colour_scatter(pcabp, clus10)

pca_50 = PCA(n_components=50)
pcabp2 = pca_50.fit_transform(bpmodmat)
print ('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
print(pcabp2.shape)
#print(pcabp)
tsnebp2 = TSNE(n_components=2).fit_transform(pcabp2)
print(tsnebp2.shape)
plt.figure('PCA TSNE Scatter plot')
#print(tsnebp2[:,0], tsnebp2[:,1])
plt.scatter(tsnebp2[:,0], tsnebp2[:,1])
plt.title('PCA TSNE Scatter plot')
plt.xlabel('x')
plt.ylabel('y')

colour_scatter(tsnebp2,clus2)
colour_scatter(tsnebp2,clus10)
############################ Plots #################################

plt.figure('Left Ventricular Pressure Peak Morphology based anomalies')
markbp = in_bp
plt.plot(numbers, bpcopy,  '-gx', markevery=markbp)
nin_bp = list(range(0,len(in_bp)))
for x, y in zip(nin_bp, in_bp):
    if x in anomlist2:
        plt.text(y, bpcopy[y], str(x), color="red", fontsize=10)
    else:
        plt.text(y, bpcopy[y], str(x), color="green", fontsize=10)

#plt.show()


end = time.time()
print("Execution time: ", (end - start))
plt.show()