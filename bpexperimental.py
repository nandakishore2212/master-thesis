import pandas as pd
import peakutils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from dtaidistance.dtaidistance import dtw
import time
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree, cut_tree
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform
import sys
#sys.setrecursionlimit(10000)
np.set_printoptions(threshold=sys.maxsize)

############# Getting data from CSV file ####################
#Extracting data - Each variable is a column in the file
start= time.time()
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

#Declaring lists to store the alternate anomaly detection parameters
bppeaklistlen = len(in_bp)
bppeaklist = np.zeros((bppeaklistlen,1))
bppeaklistavg = np.zeros((bppeaklistlen-1,1))
bppeaklistdiff = np.zeros((bppeaklistlen-1,1))
print(bppeaklist.shape)
for i in range(len(in_bp)):
    bppeaklist[i] = bp[in_bp[i]]   #Making a list of peak height value

#Average peak height and peak height difference
for i in range(len(bppeaklist)-1):
    bppeaklistavg[i] = (bppeaklist[i] + bppeaklist[i+1])/2
    bppeaklistdiff[i] = abs(bppeaklist[i] - bppeaklist[i+1])

plt.figure('Avg and diff')
plt.plot(bppeaklist, 'g', label = 'Original peaks')
plt.plot(bppeaklistavg, 'y', label = 'Average peak height' )
plt.plot(bppeaklistdiff, 'b', label = 'Peak height difference')
plt.legend()

print('Median, standard deviation peaks', np.median(bppeaklist), np.std(bppeaklist))
print('Median, standard deviation peaks average', np.median(bppeaklistavg), np.std(bppeaklistavg))
print('Median, standard deviation peaks difference', np.median(bppeaklistdiff), np.std(bppeaklistdiff))

#Values outside 3*std are marked as anomalous
bppeakanomal = []
for i in range(len(bppeaklistavg)):
    if((bppeaklistavg[i] > (np.median(bppeaklistavg) + 3* np.std(bppeaklistavg))) or (bppeaklistavg[i] < (np.median(bppeaklistavg) - 3* np.std(bppeaklistavg)))):
        bppeakanomal.append(i)
print(len(bppeakanomal), bppeakanomal)
for i in range(len(bppeaklistdiff)):
    if((bppeaklistdiff[i] > (np.median(bppeaklistdiff) + 3* np.std(bppeaklistdiff))) or (bppeaklistdiff[i] < (np.median(bppeaklistdiff) - 3* np.std(bppeaklistdiff)))):
        if(i not in bppeakanomal):#Only new anomalies are added
            bppeakanomal.append(i)
print(len(bppeakanomal), bppeakanomal)
print('Mean, std', np.median(rrint_bp), np.std(rrint_bp) )
for i in range(len(rrint_bp)):
    if((rrint_bp[i] > (np.median(rrint_bp) + 3* np.std(rrint_bp))) or (rrint_bp[i] < (np.median(rrint_bp) - 3* np.std(rrint_bp)))):
        if(i not in bppeakanomal):
            bppeakanomal.append(i)
print(len(bppeakanomal), bppeakanomal)

#Plot bp with anomalous peaks marked red
plt.figure('Blood Pressure Peak height based anomalies')
markbp = in_bp
plt.plot(numbers, bpcopy,  '-gx', markevery=markbp)
nin_bp = list(range(0,len(in_bp)))
for x, y in zip(nin_bp, in_bp):
    if x in bppeakanomal:
        plt.text(y, bpcopy[y], str(x), color="red", fontsize=10)
    else:
        plt.text(y, bpcopy[y], str(x), color="green", fontsize=10)

#Scatter plot of peak height vs peak-peak distance to see if these parameters already show clusters and clear outliers
plt.figure('Peak height vs peak-peak dist')
plt.scatter(bppeaklistavg, rrint_bp)
plt.title('Peak height vs peak-peak dist')
plt.xlabel('peak height')
plt.ylabel('peak-peak dist')

#Plot all three parameters in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rrint_bp = np.asarray(rrint_bp)
ax.scatter(bppeaklistavg, bppeaklistdiff, rrint_bp, c='b')
ax.set_xlabel('peak avg')
ax.set_ylabel('peak diff')
ax.set_zlabel('peak-peak dist')
#plt.show()

'''
################### RR interval plot ########################
rrint_bp = []
for i in range(len(in_bp)-1): #Make an array of RR interval lenths
    rrint_bp.append(in_bp[i+1]-in_bp[i])
    #print(in_bp[i+1]-in_bp[i])
#print(in_bp)
plt.figure('RR interval bp')
plt.plot(in_bp[:-1],rrint_bp)
plt.show()
print('bp: ', np.min(rrint_bp), np.max(rrint_bp), np.median(rrint_bp))


'''

clus2 = [] #These hold info on which cluster the data belong to
clus10 = []
def hycluster(X, link, metr, datatype):

    Z = linkage(X, link) #Linkage function for clustering
    cuttree = cut_tree(Z, n_clusters= [2, 10]) #Cut the dendrogram to make 2 and 10 clusters
    global clus2, clus10
    for i in cuttree:#Store cluster labels
        clus2.append(i[0])
        clus10.append(i[1])

    c, coph_dists = cophenet(Z, X) #Calculate cophenet as a benchmark for clustering
    print('Cophenet:',metr,c)
    titl = 'Hierarchical Clustering ' + datatype +','+ link +','+ metr #Make plot title based on function arguments
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
        #truncate_mode='lastp', #Show only the last p clusters
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
bpmodfull = []
bpmodmat = np.zeros((len(in_bp)-1,bpmedian))
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
#Using pdist to calculate distance between two beats at different distance metrics
XvecEuc = pdist(bpmodmat, metric = 'euclidean')
'''
XvecMink = pdist(bpmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(bpmodmat, metric = 'cityblock')
XvecSeuc = pdist(bpmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(bpmodmat, metric = 'correlation')
#XvecHamm = pdist(bpmodmat, metric = 'hamming')
#XvecMah = pdist(bpmodmat, metric = 'mahalanobis', VI=None)
'''

#Calling the hycluster function
hycluster(XvecEuc,'average','euclidean','bp')
'''
hycluster(XvecMink,'average','Minkowski','bp')
hycluster(XvecSeuc,'average','Standardised Euclidean','bp')
hycluster(XvecCity,'average','City Block','bp')
hycluster(XvecCorr,'average','Correlation','bp')
#hycluster(XvecHamm,'average','Hamming','bp')
#hycluster(XvecMah,'average','Mahalanobis','bp')
'''
print('Peak list length', len(bppeaklistavg))
anomlist2 =  []
for i in range(len(bppeaklistavg)):
    if clus2[i] == 1:
        anomlist2.append(i)
print('Number of morph anomalies', len(anomlist2))

#Identify subclusters in regular and anomalousc clusters
unique, counts = np.unique(clus10, return_counts=True)
clus10goodcounts = []
clus10anomcounts = []

print(unique,counts)
clus10goodlist =[]
clus10anomlist = []
clus10goodbeatlist = []
clus10anombeatlist = []
print('bpmodmat', bpmodmat.shape)
for i in range(len(unique)):
    temp = []#Array for beat numbers
    tempbeats = []#Array for fu
    for j in range(len(clus10)):
        if clus10[j] == unique[i]:
            temp.append(j)
            tempbeats.append(np.array(bpmodmat[j]))
    if(set(temp)<= set(anomlist2)):
        clus10anomlist.append(temp)
        clus10anombeatlist.append(tempbeats)
        clus10anomcounts.append(counts[i])
    else:
        clus10goodlist.append(temp)
        clus10goodbeatlist.append(tempbeats)
        clus10goodcounts.append(counts[i])

#print('Good list', clus10goodlist)
#print('Anom list', clus10anomlist)

#clus10goodbeatlist = np.array(clus10goodbeatlist)
#clus10anombeatlist = np.array(clus10anombeatlist, ndmin = 2)

print('Good list', clus10goodbeatlist)
#print('List shape', clus10goodbeatlist.shape )
#mediangoodlist = []
plt.figure('Median of anomalous beats')
plt.title('Median of anomalous beats')
medanombp = []
for i in range(len(clus10anombeatlist)):
    temp = np.median(clus10anombeatlist[i], axis = 0)
    medanombp.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Median of regular beats')
plt.title('Median of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.median(clus10goodbeatlist[i], axis = 0) #Taking median beat as a representative beat
    c = [rrint_bp[index] for index in clus10goodlist[i]]
    plt.plot(temp, label = [clus10goodcounts[i], 'median rr:',int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                            int(np.std(c))])

plt.plot(np.median(medanombp,axis=0), label = 'median anomalous beat')
plt.legend()

plt.figure('Mean of anomalous beats')
plt.title('Mean of anomalous beats')
meananombp =[]
for i in range(len(clus10anombeatlist)):
    temp = np.mean(clus10anombeatlist[i], axis = 0)
    meananombp.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Mean of regular beats')
plt.title('Mean of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.mean(clus10goodbeatlist[i], axis = 0)
    c = [rrint_bp[index] for index in clus10goodlist[i]]
    plt.plot(temp, label=[clus10goodcounts[i], 'median rr:', int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                          int(np.std(c))])
plt.plot(np.mean(meananombp,axis=0), label = 'mean anomalous beat')
plt.legend()

plt.figure('Standard deviation of anomalous beats')
plt.title('Standard deviation of anomalous beats')
stdanombp = []
for i in range(len(clus10anombeatlist)):
    temp = np.std(clus10anombeatlist[i], axis = 0)
    stdanombp.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Standard deviation of regular beats')
plt.title('Standard deviation of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.std(clus10goodbeatlist[i], axis = 0)
    c = [rrint_bp[index] for index in clus10goodlist[i]]
    plt.plot(temp, label=[clus10goodcounts[i], 'median rr:', int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                          int(np.std(c))])
plt.plot(np.median(stdanombp,axis=0), label = 'median anomalous beat')
plt.legend()


####################################################################################################


'''
test = list(range(0,len(bpmodfull)))
test = np.array(test)
print(len(test), len(bpmodfull))
nmatbp = int(len(bpmodfull)/bpmedian) # Number of beats
print(nmatbp)
bpmateuc = np.zeros((nmatbp,nmatbp)) #Initialising the matrices
bpmatdtw = np.zeros((nmatbp,nmatbp))

iii = ii = 0

#Distance of each beat from every other beat is calculated using both Euclidean and DTW methods
while ii<(len(bpmodfull)):
    temp1 = bpmodfull[ii: (ii+ bpmedian-1)]
    temp1 = np.array(temp1, dtype = np.double)
    jj = jjj = 0
    #print(len(temp1), type(temp1))
    while jj < len(bpmodfull):
        temp2 = bpmodfull[jj: (jj+ bpmedian-1)]
        temp2 = np.array(temp2, dtype = np.double)
        #print(len(temp2), type(temp2))
        bpmateuc[iii,jjj] = np.linalg.norm(temp1 - temp2)
        bpmatdtw[iii, jjj] = dtw.distance_fast(temp1, temp2)
        print(iii, jjj,bpmateuc[iii,jjj])
        jj = jj + bpmedian
        jjj += 1
    ii = ii + bpmedian
    iii += 1

pd.DataFrame(bpmateuc).to_csv("bpeuc.csv")
pd.DataFrame(bpmatdtw).to_csv("bpdtw.csv")
#print(bpmateuc)

'''

#print(test)
#print(bpmodfull)

#plt.plot(test,bpmodfull)
#plt.show()
############################ Plots #################################

plt.figure('Blood Pressure Peak Morphology based anomalies')
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