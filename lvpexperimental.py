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

#Declaring lists to store the alternate anomaly detection parameters
lvppeaklistlen = len(in_lvp)
lvppeaklist = np.zeros((lvppeaklistlen,1))
lvppeaklistavg = np.zeros((lvppeaklistlen-1,1))
lvppeaklistdiff = np.zeros((lvppeaklistlen-1,1))
print(lvppeaklist.shape)
for i in range(len(in_lvp)):
    lvppeaklist[i] = lvp[in_lvp[i]]   #Making a list of peak height value

#Average peak height and peak height difference
for i in range(len(lvppeaklist)-1):
    lvppeaklistavg[i] = (lvppeaklist[i] + lvppeaklist[i+1])/2
    lvppeaklistdiff[i] = abs(lvppeaklist[i] - lvppeaklist[i+1])

plt.figure('Avg and diff')
plt.plot(lvppeaklist, 'g', label = 'Original peaks')
plt.plot(lvppeaklistavg, 'y', label = 'Average peak height' )
plt.plot(lvppeaklistdiff, 'b', label = 'Peak height difference')
plt.legend()

print('Median, standard deviation peaks', np.median(lvppeaklist), np.std(lvppeaklist))
print('Median, standard deviation peaks average', np.median(lvppeaklistavg), np.std(lvppeaklistavg))
print('Median, standard deviation peaks difference', np.median(lvppeaklistdiff), np.std(lvppeaklistdiff))

#Values outside 3*std are marked as anomalous
lvppeakanomal = []
for i in range(len(lvppeaklistavg)):
    if((lvppeaklistavg[i] > (np.median(lvppeaklistavg) + 3* np.std(lvppeaklistavg))) or (lvppeaklistavg[i] < (np.median(lvppeaklistavg) - 3* np.std(lvppeaklistavg)))):
        lvppeakanomal.append(i)
print(len(lvppeakanomal), lvppeakanomal)
for i in range(len(lvppeaklistdiff)):
    if((lvppeaklistdiff[i] > (np.median(lvppeaklistdiff) + 3* np.std(lvppeaklistdiff))) or (lvppeaklistdiff[i] < (np.median(lvppeaklistdiff) - 3* np.std(lvppeaklistdiff)))):
        if(i not in lvppeakanomal):#Only new anomalies are added
            lvppeakanomal.append(i)
print(len(lvppeakanomal), lvppeakanomal)
print('Mean, std', np.median(rrint_lvp), np.std(rrint_lvp) )
for i in range(len(rrint_lvp)):
    if((rrint_lvp[i] > (np.median(rrint_lvp) + 3* np.std(rrint_lvp))) or (rrint_lvp[i] < (np.median(rrint_lvp) - 3* np.std(rrint_lvp)))):
        if(i not in lvppeakanomal):
            lvppeakanomal.append(i)
print(len(lvppeakanomal), lvppeakanomal)

#Plot LVP with anomalous peaks marked red
plt.figure('Left Ventricular Pressure Peak height based anomalies')
marklvp = in_lvp
plt.plot(numbers, lvpcopy,  '-gx', markevery=marklvp)
nin_lvp = list(range(0,len(in_lvp)))
for x, y in zip(nin_lvp, in_lvp):
    if x in lvppeakanomal:
        plt.text(y, lvpcopy[y], str(x), color="red", fontsize=10)
    else:
        plt.text(y, lvpcopy[y], str(x), color="green", fontsize=10)

#Scatter plot of peak height vs peak-peak distance to see if these parameters already show clusters and clear outliers
plt.figure('Peak height vs peak-peak dist')
plt.scatter(lvppeaklistavg, rrint_lvp)
plt.title('Peak height vs peak-peak dist')
plt.xlabel('peak height')
plt.ylabel('peak-peak dist')

#Plot all three parameters in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rrint_lvp = np.asarray(rrint_lvp)
ax.scatter(lvppeaklistavg, lvppeaklistdiff, rrint_lvp, c='b')
ax.set_xlabel('peak avg')
ax.set_ylabel('peak diff')
ax.set_zlabel('peak-peak dist')
#plt.show()

'''
################### RR interval plot ########################
rrint_lvp = []
for i in range(len(in_lvp)-1): #Make an array of RR interval lenths
    rrint_lvp.append(in_lvp[i+1]-in_lvp[i])
    #print(in_lvp[i+1]-in_lvp[i])
#print(in_lvp)
plt.figure('RR interval LVP')
plt.plot(in_lvp[:-1],rrint_lvp)
plt.show()
print('LVP: ', np.min(rrint_lvp), np.max(rrint_lvp), np.median(rrint_lvp))


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

################################LVP ######################################################
############################ Compressing each RR-interval to the median length #########################

lvpmedian = int(np.median(rrint_lvp)) #returns the median length of all the intervals
lvpmodfull = []
lvpmodmat = np.zeros((len(in_lvp)-1,lvpmedian))
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
#Using pdist to calculate distance between two beats at different distance metrics
XvecEuc = pdist(lvpmodmat, metric = 'euclidean')
'''
XvecMink = pdist(lvpmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(lvpmodmat, metric = 'cityblock')
XvecSeuc = pdist(lvpmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(lvpmodmat, metric = 'correlation')
#XvecHamm = pdist(lvpmodmat, metric = 'hamming')
#XvecMah = pdist(lvpmodmat, metric = 'mahalanobis', VI=None)
'''

#Calling the hycluster function
hycluster(XvecEuc,'average','euclidean','lvp')
'''
hycluster(XvecMink,'average','Minkowski','lvp')
hycluster(XvecSeuc,'average','Standardised Euclidean','lvp')
hycluster(XvecCity,'average','City Block','lvp')
hycluster(XvecCorr,'average','Correlation','lvp')
#hycluster(XvecHamm,'average','Hamming','lvp')
#hycluster(XvecMah,'average','Mahalanobis','lvp')
'''
print('Peak list length', len(lvppeaklistavg))
anomlist2 =  []
for i in range(len(lvppeaklistavg)):
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
print('LVPmodmat', lvpmodmat.shape)
for i in range(len(unique)):
    temp = []#Array for beat numbers
    tempbeats = []#Array for fu
    for j in range(len(clus10)):
        if clus10[j] == unique[i]:
            temp.append(j)
            tempbeats.append(np.array(lvpmodmat[j]))
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
medanomlvp = []
for i in range(len(clus10anombeatlist)):
    temp = np.median(clus10anombeatlist[i], axis = 0)
    medanomlvp.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Median of regular beats')
plt.title('Median of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.median(clus10goodbeatlist[i], axis = 0) #Taking median beat as a representative beat
    c = [rrint_lvp[index] for index in clus10goodlist[i]]
    plt.plot(temp, label = [clus10goodcounts[i], 'median rr:',int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                            int(np.std(c))])

plt.plot(np.median(medanomlvp,axis=0), label = 'median anomalous beat')
plt.legend()

plt.figure('Mean of anomalous beats')
plt.title('Mean of anomalous beats')
meananomlvp =[]
for i in range(len(clus10anombeatlist)):
    temp = np.mean(clus10anombeatlist[i], axis = 0)
    meananomlvp.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Mean of regular beats')
plt.title('Mean of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.mean(clus10goodbeatlist[i], axis = 0)
    c = [rrint_lvp[index] for index in clus10goodlist[i]]
    plt.plot(temp, label=[clus10goodcounts[i], 'median rr:', int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                          int(np.std(c))])
plt.plot(np.mean(meananomlvp,axis=0), label = 'mean anomalous beat')
plt.legend()

plt.figure('Standard deviation of anomalous beats')
plt.title('Standard deviation of anomalous beats')
stdanomlvp = []
for i in range(len(clus10anombeatlist)):
    temp = np.std(clus10anombeatlist[i], axis = 0)
    stdanomlvp.append(temp)
    plt.plot(temp, label = clus10anomcounts[i])
plt.legend()

plt.figure('Standard deviation of regular beats')
plt.title('Standard deviation of regular beats')
for i in range(len(clus10goodbeatlist)):
    temp = np.std(clus10goodbeatlist[i], axis = 0)
    c = [rrint_lvp[index] for index in clus10goodlist[i]]
    plt.plot(temp, label=[clus10goodcounts[i], 'median rr:', int(np.median(c)), 'mean rr:', int(np.mean(c)), 'std rr:',
                          int(np.std(c))])
plt.plot(np.median(stdanomlvp,axis=0), label = 'median anomalous beat')
plt.legend()


####################################################################################################


'''
test = list(range(0,len(lvpmodfull)))
test = np.array(test)
print(len(test), len(lvpmodfull))
nmatlvp = int(len(lvpmodfull)/lvpmedian) # Number of beats
print(nmatlvp)
lvpmateuc = np.zeros((nmatlvp,nmatlvp)) #Initialising the matrices
lvpmatdtw = np.zeros((nmatlvp,nmatlvp))

iii = ii = 0

#Distance of each beat from every other beat is calculated using both Euclidean and DTW methods
while ii<(len(lvpmodfull)):
    temp1 = lvpmodfull[ii: (ii+ lvpmedian-1)]
    temp1 = np.array(temp1, dtype = np.double)
    jj = jjj = 0
    #print(len(temp1), type(temp1))
    while jj < len(lvpmodfull):
        temp2 = lvpmodfull[jj: (jj+ lvpmedian-1)]
        temp2 = np.array(temp2, dtype = np.double)
        #print(len(temp2), type(temp2))
        lvpmateuc[iii,jjj] = np.linalg.norm(temp1 - temp2)
        lvpmatdtw[iii, jjj] = dtw.distance_fast(temp1, temp2)
        print(iii, jjj,lvpmateuc[iii,jjj])
        jj = jj + lvpmedian
        jjj += 1
    ii = ii + lvpmedian
    iii += 1

pd.DataFrame(lvpmateuc).to_csv("Lvpeuc.csv")
pd.DataFrame(lvpmatdtw).to_csv("Lvpdtw.csv")
#print(lvpmateuc)

'''

#print(test)
#print(lvpmodfull)

#plt.plot(test,lvpmodfull)
#plt.show()
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