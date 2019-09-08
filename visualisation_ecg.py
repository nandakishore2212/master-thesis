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
ecg = bayerdata[0:3600000][3]

numbers = list(range(0,3600000))

#Converting to numpy array for easier processing
#timearray = np.array(timearray)
ecg = ecgcopy= np.array(ecg)





##################### Finding peaks  ##############

### ECG ###
in_ecg =[]
ecg_thres =0.5 * (np.max(ecg)- np.min(ecg))+np.min(ecg)
for i in range(3600):
    if i != 0:
        ecg_sm = ecg[1000*i-1:1000*(i+1)+1]
        in_ecg_sm = peakutils.indexes(ecg_sm, thres=0.7, min_dist=200)
        in_ecg_abs = peakutils.indexes(ecg_sm, thres=ecg_thres, min_dist=200, thres_abs=True)
        in_ecg_sm = list(set(in_ecg_sm) & set(in_ecg_abs))
        in_ecg_sm.sort()
        in_ecg_sm_adj = [element + i * 1000 - 1 for element in in_ecg_sm]

    else:
        ecg_sm = ecg[1000 * i :1000 * (i + 1) + 1]
        in_ecg_sm = peakutils.indexes(ecg_sm, thres=0.7, min_dist=200)
        in_ecg_abs = peakutils.indexes(ecg_sm, thres=ecg_thres, min_dist=200, thres_abs=True)
        in_ecg_sm = list(set(in_ecg_sm) & set(in_ecg_abs))
        in_ecg_sm.sort()
        in_ecg_sm_adj = [element + i*1000 for element in in_ecg_sm]

    if in_ecg and in_ecg_sm_adj:
        if (in_ecg_sm_adj[0] - in_ecg[-1] < 200):
            if (ecg[in_ecg_sm_adj[0]] > ecg[in_ecg[-1]]):
                in_ecg.pop()
            elif (ecg[in_ecg[-1]] > ecg[in_ecg_sm_adj[0]]):
                in_ecg_sm_adj = in_ecg_sm_adj[1:]
            else:
                in_ecg_sm_adj[0] = int((in_ecg_sm_adj[0] + in_ecg[-1]) / 2)
                in_ecg.pop()

    in_ecg = in_ecg + in_ecg_sm_adj
#in_ecg = in_ecg.tolist()
print(len(in_ecg))


rrint_ecg = []
for i in range(len(in_ecg)-1): #Make an array of RR interval lengths
    rrint_ecg.append(in_ecg[i+1]-in_ecg[i])






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

################################ecg ######################################################
############################ Compressing each RR-interval to the median length #########################

ecgmedian = int(np.median(rrint_ecg)) #returns the median length of all the intervals
#print(ecgmedian)
#print('ecg beat number', len(in_ecg))
ecgmodfull = []
ecgmodmat = np.zeros((len(in_ecg)-1,ecgmedian))
#print('Mat shape', ecgmodmat.shape)
ecgmodfull = np.array(ecgmodfull)
for i in range(len(in_ecg)-1):
    nn = in_ecg[i+1]-in_ecg[i] #length of a beat
    ecgcomp1 = np.linspace(0, nn-1, num=nn, endpoint=True) #Defining an x axis with as many points as the beat length
    ecgres1 = ecg[in_ecg[i]:in_ecg[i+1]] #The actual data
    f1 = interp1d(ecgcomp1, ecgres1, kind='nearest') #A function is defined that interpolates the x axis to the nearest
                                                    #point in the dataset. Right now the two arrays are 1 to 1 (equal size)

    ecgxnew = np.linspace(0, nn-1, num= int(ecgmedian), endpoint=True) #Now we declare a new X-axis with the same length
                #but the number of points in the axis is equal to the median length. If the median length is lower than
                #axis, it'll be sparsely populated and if higher, more densely.

    ecgmod = f1(ecgxnew) #Using the same functioned on the new axis finds corresponding points in new axis. Meaning the
            #beat is effectively compressed or stretched.
    #print('Types', type(ecgmod), type(ecgmodfull))
    ecgmodfull = np.append(ecgmodfull,ecgmod)  #Full array with all beats of same median length
    ecgmodmat[i] = ecgmod
    #print(len(ecgmod))
    #plt.plot(ecgcomp1, ecgres1, '-', ecgxnew, ecgmod, 'o')
    #plt.show()

    #print('Len:', len(ecgcomp1), len(ecgres1))
    #break
    #rrint_ecg.append(in_ecg[i+1]-in_ecg[i])
'''for i in ecgpeakanomal:
    ecgmodmat[ecgmedian*i]= ecgpeaklistavg[ecgmedian*i]
    np.delete(ecgmodmat, [ecgmedian*i:(ecgmedian*(i+1))])
'''
#print(ecgmodmat.shape)
XvecEuc = pdist(ecgmodmat, metric = 'euclidean')
'''
XvecMink = pdist(ecgmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(ecgmodmat, metric = 'cityblock')
XvecSeuc = pdist(ecgmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(ecgmodmat, metric = 'correlation')
#XvecHamm = pdist(ecgmodmat, metric = 'hamming')
#XvecMah = pdist(ecgmodmat, metric = 'mahalanobis', VI=None)
'''


hycluster(XvecEuc,'average','euclidean','ecg')

'''
hycluster(XvecMink,'average','Minkowski','ecg')
hycluster(XvecSeuc,'average','Standardised Euclidean','ecg')
hycluster(XvecCity,'average','City Block','ecg')
hycluster(XvecCorr,'average','Correlation','ecg')
#hycluster(XvecHamm,'average','Hamming','ecg')
#hycluster(XvecMah,'average','Mahalanobis','ecg')
'''
print('Peak list length', len(rrint_ecg))
anomlist2 =  []
for i in range(len(rrint_ecg)):
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
tsneecg = TSNE(n_components=2).fit_transform(ecgmodmat)
print(tsneecg.shape)
#print(tsneecg)
plt.figure('TSNE Scatter plot')
print(tsneecg[:,0], tsneecg[:,1])
plt.scatter(tsneecg[:,0], tsneecg[:,1])
plt.title('TSNE Scatter plot')
plt.xlabel('x')
plt.ylabel('y')

clus2 = np.asarray(clus2)
clus10 = np.asarray(clus10)
print('Shapes', tsneecg.shape, clus2.shape)
colour_scatter(tsneecg, clus2)
colour_scatter(tsneecg, clus10)

pca = PCA(n_components=2)
pcaecg = pca.fit_transform(ecgmodmat)
print ('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
print(pcaecg.shape)
#print(pcaecg)
plt.figure('PCA Scatter plot')
#print(pcaecg[:,0], pcaecg[:,1])
plt.scatter(pcaecg[:,0], pcaecg[:,1])
plt.title('PCA Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
colour_scatter(pcaecg,clus2)
colour_scatter(pcaecg, clus10)

pca_50 = PCA(n_components=50)
pcaecg2 = pca_50.fit_transform(ecgmodmat)
print ('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
print(pcaecg2.shape)
#print(pcaecg)
tsneecg2 = TSNE(n_components=2).fit_transform(pcaecg2)
print(tsneecg2.shape)
plt.figure('PCA TSNE Scatter plot')
#print(tsneecg2[:,0], tsneecg2[:,1])
plt.scatter(tsneecg2[:,0], tsneecg2[:,1])
plt.title('PCA TSNE Scatter plot')
plt.xlabel('x')
plt.ylabel('y')

colour_scatter(tsneecg2,clus2)
colour_scatter(tsneecg2,clus10)
############################ Plots #################################

plt.figure('Left Ventricular Pressure Peak Morphology based anomalies')
markecg = in_ecg
plt.plot(numbers, ecgcopy,  '-gx', markevery=markecg)
nin_ecg = list(range(0,len(in_ecg)))
for x, y in zip(nin_ecg, in_ecg):
    if x in anomlist2:
        plt.text(y, ecgcopy[y], str(x), color="red", fontsize=10)
    else:
        plt.text(y, ecgcopy[y], str(x), color="green", fontsize=10)

#plt.show()


end = time.time()
print("Execution time: ", (end - start))
plt.show()