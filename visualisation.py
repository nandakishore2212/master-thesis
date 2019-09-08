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
lvp = bayerdata[0:3600000][1]

numbers = list(range(0,3600000))

#Converting to numpy array for easier processing
#timearray = np.array(timearray)
lvp = lvpcopy= np.array(lvp)


#Savitzky-Golay filter is applied so that mini-peaks due to fluctuations are smoothed out
lvp = savgol_filter(lvp, 51, 3)


#Code to split a sequence into num pieces, find the min and max of each and average it.
def split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    min = 0
    max = 0
    while last < len(seq):
        piece= seq[int(last):int(last + avg)]
        min += np.min(piece)
        max+= np.max(piece)
        last += avg
    min /= num
    max/= num
    return min, max

##################### Finding peaks  ##############

### LVP ###
in_lvp =[]
minlvp, maxlvp = split(lvp,20)
#lvp_thres =0.5 * (np.max(lvp)- np.min(lvp))+np.min(lvp)
lvp_thres =0.5 * (maxlvp- minlvp)+minlvp #Conservative min threshold for peak height
print(lvp_thres)
for i in range(3600):
    #peakutils detects peaks using derivatives. So peaks at the ends of the signal cannot be identified. Hence the
    #signal is divided as 0-1001, 1000-2001, 2000-3001, etc. Signal is divided and checked as there are fluctuations
    #and a global min and max cannot set accurate thresholds for peak detection.
    if i != 0:
        lvp_sm = lvp[1000*i-1:1000*(i+1)+1]
        #Peaks are found using two criteria: 1. Greater than half of the mid in the sequence, 2. Higher than absolute
        #threshold. Only the peaks fulfilling both criteria are valid peaks
        in_lvp_sm = peakutils.indexes(lvp_sm, thres=0.5, min_dist=200)
        in_lvp_abs = peakutils.indexes(lvp_sm, thres=lvp_thres, min_dist=200, thres_abs=True)
        in_lvp_sm = list(set(in_lvp_sm) & set(in_lvp_abs))
        in_lvp_sm.sort() #set operations can mess up the order, so sorting is required
        in_lvp_sm_adj = [element + i * 1000 - 1 for element in in_lvp_sm] # Setting the absolute peak locations
    else:
        lvp_sm = lvp[1000 * i :1000 * (i + 1) + 1]
        in_lvp_sm = peakutils.indexes(lvp_sm, thres=0.5, min_dist=200)
        in_lvp_abs = peakutils.indexes(lvp_sm, thres=lvp_thres, min_dist=200, thres_abs=True)
        in_lvp_sm = list(set(in_lvp_sm) & set(in_lvp_abs))
        in_lvp_sm.sort()
        in_lvp_sm_adj = [element + i*1000 for element in in_lvp_sm]
    #Adjacent peaks cannot be closer than 200. This is taken care of for each subsequence but it can happen that last
    #peak in one and first peak in the next are very close. So, the larger of the two peaks is chosen and the other removed
    if in_lvp and in_lvp_sm_adj:
        if(in_lvp_sm_adj[0]-in_lvp[-1] < 200):
            if(lvp[in_lvp_sm_adj[0]] > lvp[in_lvp[-1]]):
                in_lvp.pop()
            elif(lvp[in_lvp[-1]] >lvp[in_lvp_sm_adj[0]]):
                in_lvp_sm_adj = in_lvp_sm_adj[1:]
            else:
                in_lvp_sm_adj[0] = int((in_lvp_sm_adj[0]+ in_lvp[-1])/2)
                in_lvp.pop()

    in_lvp = in_lvp + in_lvp_sm_adj # final list of peaks


rrint_lvp = []
for i in range(len(in_lvp)-1): #Make an array of RR interval lengths
    rrint_lvp.append(in_lvp[i+1]-in_lvp[i])






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

################################LVP ######################################################
############################ Compressing each RR-interval to the median length #########################

lvpmedian = int(np.median(rrint_lvp)) #returns the median length of all the intervals
#print(lvpmedian)
#print('LVP beat number', len(in_lvp))
lvpmodfull = []
lvpmodmat = np.zeros((len(in_lvp)-1,lvpmedian))
#print('Mat shape', lvpmodmat.shape)
lvpmodfull = np.array(lvpmodfull)
for i in range(len(in_lvp)-1):
    nn = in_lvp[i+1]-in_lvp[i] #length of a beat
    lvpcomp1 = np.linspace(0, nn-1, num=nn, endpoint=True) #Defining an x axis with as many points as the beat length
    lvpres1 = lvp[in_lvp[i]:in_lvp[i+1]] #The actual data
    f1 = interp1d(lvpcomp1, lvpres1, kind='nearest') #A function is defined that interpolates the x axis to the nearest
                                                    #point in the dataset. Right now the two arrays are 1 to 1 (equal size)
    
    lvpxnew = np.linspace(0, nn-1, num= int(lvpmedian), endpoint=True) #Now we declare a new X-axis with the same length
                #but the number of points in the axis is equal to the median length. If the median length is lower than
                #axis, it'll be sparsely populated and if higher, more densely.

    lvpmod = f1(lvpxnew) #Using the same functioned on the new axis finds corresponding points in new axis. Meaning the
            #beat is effectively compressed or stretched.
    #print('Types', type(lvpmod), type(lvpmodfull))
    lvpmodfull = np.append(lvpmodfull,lvpmod)  #Full array with all beats of same median length
    lvpmodmat[i] = lvpmod
    #print(len(lvpmod))
    #plt.plot(lvpcomp1, lvpres1, '-', lvpxnew, lvpmod, 'o')
    #plt.show()

    #print('Len:', len(lvpcomp1), len(lvpres1))
    #break
    #rrint_lvp.append(in_lvp[i+1]-in_lvp[i])
'''for i in lvppeakanomal:
    lvpmodmat[lvpmedian*i]= lvppeaklistavg[lvpmedian*i]
    np.delete(lvpmodmat, [lvpmedian*i:(lvpmedian*(i+1))])
'''
#print(lvpmodmat.shape)
XvecEuc = pdist(lvpmodmat, metric = 'euclidean')
'''
XvecMink = pdist(lvpmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(lvpmodmat, metric = 'cityblock')
XvecSeuc = pdist(lvpmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(lvpmodmat, metric = 'correlation')
#XvecHamm = pdist(lvpmodmat, metric = 'hamming')
#XvecMah = pdist(lvpmodmat, metric = 'mahalanobis', VI=None)
'''


hycluster(XvecEuc,'average','euclidean','lvp')

'''
hycluster(XvecMink,'average','Minkowski','lvp')
hycluster(XvecSeuc,'average','Standardised Euclidean','lvp')
hycluster(XvecCity,'average','City Block','lvp')
hycluster(XvecCorr,'average','Correlation','lvp')
#hycluster(XvecHamm,'average','Hamming','lvp')
#hycluster(XvecMah,'average','Mahalanobis','lvp')
'''
print('Peak list length', len(rrint_lvp))
anomlist2 =  []
for i in range(len(rrint_lvp)):
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
tsnelvp = TSNE(n_components=2).fit_transform(lvpmodmat)
print(tsnelvp.shape)
#print(tsnelvp)
plt.figure('TSNE Scatter plot')
print(tsnelvp[:,0], tsnelvp[:,1])
plt.scatter(tsnelvp[:,0], tsnelvp[:,1])
plt.title('TSNE Scatter plot')
plt.xlabel('x')
plt.ylabel('y')

clus2 = np.asarray(clus2)
clus10 = np.asarray(clus10)
print('Shapes', tsnelvp.shape, clus2.shape)
colour_scatter(tsnelvp, clus2)
colour_scatter(tsnelvp, clus10)

pca = PCA(n_components=2)
pcalvp = pca.fit_transform(lvpmodmat)
print ('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
print(pcalvp.shape)
#print(pcalvp)
plt.figure('PCA Scatter plot')
#print(pcalvp[:,0], pcalvp[:,1])
plt.scatter(pcalvp[:,0], pcalvp[:,1])
plt.title('PCA Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
colour_scatter(pcalvp,clus2)
colour_scatter(pcalvp, clus10)

pca_50 = PCA(n_components=50)
pcalvp2 = pca_50.fit_transform(lvpmodmat)
print ('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
print(pcalvp2.shape)
#print(pcalvp)
tsnelvp2 = TSNE(n_components=2).fit_transform(pcalvp2)
print(tsnelvp2.shape)
plt.figure('PCA TSNE Scatter plot')
#print(tsnelvp2[:,0], tsnelvp2[:,1])
plt.scatter(tsnelvp2[:,0], tsnelvp2[:,1])
plt.title('PCA TSNE Scatter plot')
plt.xlabel('x')
plt.ylabel('y')

colour_scatter(tsnelvp2,clus2)
colour_scatter(tsnelvp2,clus10)
############################ Plots #################################

plt.figure('Left Ventricular Pressure Peak Morphology based anomalies')
marklvp = in_lvp
plt.plot(numbers, lvpcopy,  '-gx', markevery=marklvp)
nin_lvp = list(range(0,len(in_lvp)))
for x, y in zip(nin_lvp, in_lvp):
    if x in anomlist2:
        plt.text(y, lvpcopy[y], str(x), color="red", fontsize=10)
    else:
        plt.text(y, lvpcopy[y], str(x), color="green", fontsize=10)

#plt.show()


end = time.time()
print("Execution time: ", (end - start))
plt.show()