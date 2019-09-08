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

#Declaring lists to store the alternate anomaly detection parameters
ecgpeaklistlen = len(in_ecg)
ecgpeaklist = np.zeros((ecgpeaklistlen,1))
ecgpeaklistavg = np.zeros((ecgpeaklistlen-1,1))
ecgpeaklistdiff = np.zeros((ecgpeaklistlen-1,1))
print(ecgpeaklist.shape)
for i in range(len(in_ecg)):
    ecgpeaklist[i] = ecg[in_ecg[i]]   #Making a list of peak height value

#Average peak height and peak height difference
for i in range(len(ecgpeaklist)-1):
    ecgpeaklistavg[i] = (ecgpeaklist[i] + ecgpeaklist[i+1])/2
    ecgpeaklistdiff[i] = abs(ecgpeaklist[i] - ecgpeaklist[i+1])

plt.figure('Avg and diff')
plt.plot(ecgpeaklist, 'g', label = 'Original peaks')
plt.plot(ecgpeaklistavg, 'y', label = 'Average peak height' )
plt.plot(ecgpeaklistdiff, 'b', label = 'Peak height difference')
plt.legend()

print('Median, standard deviation peaks', np.median(ecgpeaklist), np.std(ecgpeaklist))
print('Median, standard deviation peaks average', np.median(ecgpeaklistavg), np.std(ecgpeaklistavg))
print('Median, standard deviation peaks difference', np.median(ecgpeaklistdiff), np.std(ecgpeaklistdiff))

#Values outside 3*std are marked as anomalous
ecgpeakanomal = []
for i in range(len(ecgpeaklistavg)):
    if((ecgpeaklistavg[i] > (np.median(ecgpeaklistavg) + 3* np.std(ecgpeaklistavg))) or (ecgpeaklistavg[i] < (np.median(ecgpeaklistavg) - 3* np.std(ecgpeaklistavg)))):
        ecgpeakanomal.append(i)
print(len(ecgpeakanomal), ecgpeakanomal)
for i in range(len(ecgpeaklistdiff)):
    if((ecgpeaklistdiff[i] > (np.median(ecgpeaklistdiff) + 3* np.std(ecgpeaklistdiff))) or (ecgpeaklistdiff[i] < (np.median(ecgpeaklistdiff) - 3* np.std(ecgpeaklistdiff)))):
        if(i not in ecgpeakanomal):#Only new anomalies are added
            ecgpeakanomal.append(i)
print(len(ecgpeakanomal), ecgpeakanomal)
print('Mean, std', np.median(rrint_ecg), np.std(rrint_ecg) )
for i in range(len(rrint_ecg)):
    if((rrint_ecg[i] > (np.median(rrint_ecg) + 3* np.std(rrint_ecg))) or (rrint_ecg[i] < (np.median(rrint_ecg) - 3* np.std(rrint_ecg)))):
        if(i not in ecgpeakanomal):
            ecgpeakanomal.append(i)
print(len(ecgpeakanomal), ecgpeakanomal)

#Plot ecg with anomalous peaks marked red
plt.figure('Left Ventricular Pressure Peak height based anomalies')
markecg = in_ecg
plt.plot(numbers, ecgcopy,  '-gx', markevery=markecg)
nin_ecg = list(range(0,len(in_ecg)))
for x, y in zip(nin_ecg, in_ecg):
    if x in ecgpeakanomal:
        plt.text(y, ecgcopy[y], str(x), color="red", fontsize=10)
    else:
        plt.text(y, ecgcopy[y], str(x), color="green", fontsize=10)

#Scatter plot of peak height vs peak-peak distance to see if these parameters already show clusters and clear outliers
plt.figure('Peak height vs peak-peak dist')
plt.scatter(ecgpeaklistavg, rrint_ecg)
plt.title('Peak height vs peak-peak dist')
plt.xlabel('peak height')
plt.ylabel('peak-peak dist')

#Plot all three parameters in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rrint_ecg = np.asarray(rrint_ecg)
ax.scatter(ecgpeaklistavg, ecgpeaklistdiff, rrint_ecg, c='b')
ax.set_xlabel('peak avg')
ax.set_ylabel('peak diff')
ax.set_zlabel('peak-peak dist')
#plt.show()

'''
################### RR interval plot ########################
rrint_ecg = []
for i in range(len(in_ecg)-1): #Make an array of RR interval lenths
    rrint_ecg.append(in_ecg[i+1]-in_ecg[i])
    #print(in_ecg[i+1]-in_ecg[i])
#print(in_ecg)
plt.figure('RR interval ecg')
plt.plot(in_ecg[:-1],rrint_ecg)
plt.show()
print('ecg: ', np.min(rrint_ecg), np.max(rrint_ecg), np.median(rrint_ecg))


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

################################ecg ######################################################
############################ Compressing each RR-interval to the median length #########################

ecgmedian = int(np.median(rrint_ecg)) #returns the median length of all the intervals
ecgmodfull = []
ecgmodmat = np.zeros((len(in_ecg)-1,ecgmedian))
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
#Using pdist to calculate distance between two beats at different distance metrics
XvecEuc = pdist(ecgmodmat, metric = 'euclidean')
'''
XvecMink = pdist(ecgmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(ecgmodmat, metric = 'cityblock')
XvecSeuc = pdist(ecgmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(ecgmodmat, metric = 'correlation')
#XvecHamm = pdist(ecgmodmat, metric = 'hamming')
#XvecMah = pdist(ecgmodmat, metric = 'mahalanobis', VI=None)
'''

#Calling the hycluster function
hycluster(XvecEuc,'average','euclidean','ecg')
'''
hycluster(XvecMink,'average','Minkowski','ecg')
hycluster(XvecSeuc,'average','Standardised Euclidean','ecg')
hycluster(XvecCity,'average','City Block','ecg')
hycluster(XvecCorr,'average','Correlation','ecg')
#hycluster(XvecHamm,'average','Hamming','ecg')
#hycluster(XvecMah,'average','Mahalanobis','ecg')
'''
print('Peak list length', len(ecgpeaklistavg))
anomlist2 =  []
for i in range(len(ecgpeaklistavg)):
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
print('ecgmodmat', ecgmodmat.shape)
for i in range(len(unique)):
    temp = []#Array for beat numbers
    tempbeats = []#Array for fu
    for j in range(len(clus10)):
        if clus10[j] == unique[i]:
            temp.append(j)
            tempbeats.append(np.array(ecgmodmat[j]))
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
medanomecg = []
for i in range(len(clus10anombeatlist)):
    temp = np.median(clus10anombeatlist[i], axis = 0)
    medanomecg.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Median of regular beats')
plt.title('Median of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.median(clus10goodbeatlist[i], axis = 0) #Taking median beat as a representative beat
    c = [rrint_ecg[index] for index in clus10goodlist[i]]
    plt.plot(temp, label = [clus10goodcounts[i], 'median rr:',int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                            int(np.std(c))])

plt.plot(np.median(medanomecg,axis=0), label = 'median anomalous beat')
plt.legend()

plt.figure('Mean of anomalous beats')
plt.title('Mean of anomalous beats')
meananomecg =[]
for i in range(len(clus10anombeatlist)):
    temp = np.mean(clus10anombeatlist[i], axis = 0)
    meananomecg.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Mean of regular beats')
plt.title('Mean of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.mean(clus10goodbeatlist[i], axis = 0)
    c = [rrint_ecg[index] for index in clus10goodlist[i]]
    plt.plot(temp, label=[clus10goodcounts[i], 'median rr:', int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                          int(np.std(c))])
plt.plot(np.mean(meananomecg,axis=0), label = 'mean anomalous beat')
plt.legend()

plt.figure('Standard deviation of anomalous beats')
plt.title('Standard deviation of anomalous beats')
stdanomecg = []
for i in range(len(clus10anombeatlist)):
    temp = np.std(clus10anombeatlist[i], axis = 0)
    stdanomecg.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Standard deviation of regular beats')
plt.title('Standard deviation of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.std(clus10goodbeatlist[i], axis = 0)
    c = [rrint_ecg[index] for index in clus10goodlist[i]]
    plt.plot(temp, label=[clus10goodcounts[i], 'median rr:', int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                          int(np.std(c))])
plt.plot(np.median(stdanomecg,axis=0), label = 'median anomalous beat')
plt.legend()


####################################################################################################


'''
test = list(range(0,len(ecgmodfull)))
test = np.array(test)
print(len(test), len(ecgmodfull))
nmatecg = int(len(ecgmodfull)/ecgmedian) # Number of beats
print(nmatecg)
ecgmateuc = np.zeros((nmatecg,nmatecg)) #Initialising the matrices
ecgmatdtw = np.zeros((nmatecg,nmatecg))

iii = ii = 0

#Distance of each beat from every other beat is calculated using both Euclidean and DTW methods
while ii<(len(ecgmodfull)):
    temp1 = ecgmodfull[ii: (ii+ ecgmedian-1)]
    temp1 = np.array(temp1, dtype = np.double)
    jj = jjj = 0
    #print(len(temp1), type(temp1))
    while jj < len(ecgmodfull):
        temp2 = ecgmodfull[jj: (jj+ ecgmedian-1)]
        temp2 = np.array(temp2, dtype = np.double)
        #print(len(temp2), type(temp2))
        ecgmateuc[iii,jjj] = np.linalg.norm(temp1 - temp2)
        ecgmatdtw[iii, jjj] = dtw.distance_fast(temp1, temp2)
        print(iii, jjj,ecgmateuc[iii,jjj])
        jj = jj + ecgmedian
        jjj += 1
    ii = ii + ecgmedian
    iii += 1

pd.DataFrame(ecgmateuc).to_csv("ecgeuc.csv")
pd.DataFrame(ecgmatdtw).to_csv("ecgdtw.csv")
#print(ecgmateuc)

'''

#print(test)
#print(ecgmodfull)

#plt.plot(test,ecgmodfull)
#plt.show()
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