import pandas as pd
import peakutils
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from dtaidistance.dtaidistance import dtw
import time
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform
#import sys
#sys.setrecursionlimit(10000)

############# Getting data from CSV file ####################
#Extracting data - Each variable is a column in the file
start= time.time()
bayerdata = pd.read_csv('D:\\acads\\semester4\\thesis\\BAYER data\\T521-48-R.csv', sep = ' ', header = None)
#timearray = bayerdata[0:3600000][0]
lvp = bayerdata[0:3600000][1]
#print(lvp)
bp = bayerdata[0:3600000][2]
ecg = bayerdata[0:3600000][3]
numbers = list(range(0,3600000))

#Converting to numpy array for easier processing
#timearray = np.array(timearray)
lvp = lvpcopy= np.array(lvp)

#print(lvp)
bp = bpcopy = np.array(bp)
ecg = ecgcopy = np.array(ecg)

#Savitzky-Golay filter is applied so that mini-peaks due to fluctuations are smoothed out
lvp = savgol_filter(lvp, 51, 3)
bp = savgol_filter(bp, 101, 3)

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



#in_lvp = in_lvp.tolist()
#print(in_lvp)
#print(in_lvp)
lvppeaklistlen = len(in_lvp)

lvppeaklist = np.zeros((lvppeaklistlen,1))
lvppeaklistavg = np.zeros((lvppeaklistlen-1,1))
lvppeaklistdiff = np.zeros((lvppeaklistlen-1,1))
print(lvppeaklist.shape)
for i in range(len(in_lvp)):
    lvppeaklist[i] = lvp[in_lvp[i]]

#print(peaklist)

for i in range(len(lvppeaklist)-1):
    lvppeaklistavg[i] = (lvppeaklist[i] + lvppeaklist[i+1])/2
    lvppeaklistdiff[i] = abs(lvppeaklist[i] - lvppeaklist[i+1])


plt.figure('Avg and diff')
#ax = plt.subplot(111)
plt.plot(lvppeaklist, 'g', label = 'Original peaks')
plt.plot(lvppeaklistavg, 'y', label = 'Average peak height' )
plt.plot(lvppeaklistdiff, 'b', label = 'Peak height difference')
plt.legend()

#print('done')
print('Median, standard deviation peaks', np.median(lvppeaklist), np.std(lvppeaklist))
print('Median, standard deviation peaks average', np.median(lvppeaklistavg), np.std(lvppeaklistavg))
print('Median, standard deviation peaks difference', np.median(lvppeaklistdiff), np.std(lvppeaklistdiff))

lvppeakanomal = []
for i in range(len(lvppeaklistavg)):
    if((lvppeaklistavg[i] > (np.median(lvppeaklistavg) + 2* np.std(lvppeaklistavg))) or (lvppeaklistavg[i] < (np.median(lvppeaklistavg) - 3* np.std(lvppeaklistavg)))):
        lvppeakanomal.append(i)
print(len(lvppeakanomal), lvppeakanomal)
for i in range(len(lvppeaklistdiff)):
    if((lvppeaklistdiff[i] > (np.median(lvppeaklistdiff) + 2* np.std(lvppeaklistdiff))) or (lvppeaklistdiff[i] < (np.median(lvppeaklistdiff) - 3* np.std(lvppeaklistdiff)))):
        if(i not in lvppeakanomal):
            lvppeakanomal.append(i)
print(len(lvppeakanomal), lvppeakanomal)

plt.figure('Left Ventricular Pressure')
marklvp = in_lvp
plt.plot(numbers, lvpcopy,  '-gx', markevery=marklvp)
nin_lvp = list(range(0,len(in_lvp)))
for x, y in zip(nin_lvp, in_lvp):
    if x in lvppeakanomal:
        plt.text(y, lvpcopy[y], str(x), color="red", fontsize=10)
    else:
        plt.text(y, lvpcopy[y], str(x), color="green", fontsize=10)

plt.show()
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
#print("Edited bp",in_bp)
#print("unedited bp", in_bp_2)


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

rrint_bp = []
for i in range(len(in_bp)-1):
    rrint_bp.append(in_bp[i+1]-in_bp[i])
    #print(in_bp[i+1]-in_bp[i])
#print(in_bp)
plt.figure('RR interval BP')
plt.plot(in_bp[:-1],rrint_bp)
#plt.show()
print('BP: ', np.min(rrint_bp), np.max(rrint_bp), np.median(rrint_bp))


rrint_ecg = []
for i in range(len(in_ecg)-1):
    rrint_ecg.append(in_ecg[i+1]-in_ecg[i])
    #print(in_ecg[i+1]-in_ecg[i])
#print(in_ecg)
plt.figure('RR interval ECG')
plt.plot(in_ecg[:-1],rrint_ecg)
#plt.show()
print('ECG: ', np.min(rrint_ecg), np.max(rrint_ecg), np.median(rrint_ecg))
'''

def hycluster(X, link, metr, datatype):

    # generate the linkage matrix
    Z = linkage(X, link)

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
rrint_lvp = []
for i in range(len(in_lvp)-1): #Make an array of RR interval lengths
    rrint_lvp.append(in_lvp[i+1]-in_lvp[i])
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
XvecMink = pdist(lvpmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(lvpmodmat, metric = 'cityblock')
XvecSeuc = pdist(lvpmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(lvpmodmat, metric = 'correlation')
#XvecHamm = pdist(lvpmodmat, metric = 'hamming')
#XvecMah = pdist(lvpmodmat, metric = 'mahalanobis', VI=None)



hycluster(XvecEuc,'average','euclidean','lvp')
hycluster(XvecMink,'average','Minkowski','lvp')
hycluster(XvecSeuc,'average','Standardised Euclidean','lvp')
hycluster(XvecCity,'average','City Block','lvp')
hycluster(XvecCorr,'average','Correlation','lvp')
#hycluster(XvecHamm,'average','Hamming','lvp')
#hycluster(XvecMah,'average','Mahalanobis','lvp')
####################################################################################################

################################ BP ######################################################
############################ Compressing each RR-interval to the median length #########################
rrint_bp = []
for i in range(len(in_bp)-1): #Make an array of RR interval lengths
    rrint_bp.append(in_bp[i+1]-in_bp[i])
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

#print(bpmodmat.shape)
XvecEuc = pdist(bpmodmat, metric = 'euclidean')
XvecMink = pdist(bpmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(bpmodmat, metric = 'cityblock')
XvecSeuc = pdist(bpmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(bpmodmat, metric = 'correlation')
#XvecHamm = pdist(bpmodmat, metric = 'hamming')
#XvecMah = pdist(bpmodmat, metric = 'mahalanobis', VI=None)

hycluster(XvecEuc,'average','euclidean','bp')
hycluster(XvecMink,'average','Minkowski','bp')
hycluster(XvecSeuc,'average','Standardised Euclidean','bp')
hycluster(XvecCity,'average','City Block','bp')
hycluster(XvecCorr,'average','Correlation','bp')
#hycluster(XvecHamm,'average','Hamming','bp')
#hycluster(XvecMah,'average','Mahalanobis','bp')
############################################################################################

################################ ECG ######################################################
############################ Compressing each RR-interval to the median length #########################
rrint_ecg = []
for i in range(len(in_ecg)-1): #Make an array of RR interval lengths
    rrint_ecg.append(in_ecg[i+1]-in_ecg[i])
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

#print(ecgmodmat.shape)
XvecEuc = pdist(ecgmodmat, metric = 'euclidean')
XvecMink = pdist(ecgmodmat, metric = 'minkowski', p=3)
XvecCity = pdist(ecgmodmat, metric = 'cityblock')
XvecSeuc = pdist(ecgmodmat, metric = 'seuclidean', V = None)
XvecCorr = pdist(ecgmodmat, metric = 'correlation')
#XvecHamm = pdist(ecgmodmat, metric = 'hamming')
#XvecMah = pdist(ecgmodmat, metric = 'mahalanobis', VI=None)

hycluster(XvecEuc,'average','euclidean','ecg')
hycluster(XvecMink,'average','Minkowski','ecg')
hycluster(XvecSeuc,'average','Standardised Euclidean','ecg')
hycluster(XvecCity,'average','City Block','ecg')
hycluster(XvecCorr,'average','Correlation','ecg')
#hycluster(XvecHamm,'average','Hamming','ecg')
#hycluster(XvecMah,'average','Mahalanobis','ecg')
############################################################################################

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

plt.figure('Left Ventricular Pressure')
marklvp = in_lvp
plt.plot(numbers, lvpcopy,  '-gx', markevery=marklvp)
nin_lvp = list(range(0,len(in_lvp)))
for x, y in zip(nin_lvp, in_lvp):
    plt.text(y, lvpcopy[y], str(x), color="red", fontsize=12)
#plt.show()



plt.figure('Blood Pressure')
markbp = in_bp
plt.plot(numbers, bp, '-gx', markevery=markbp)
nin_bp = list(range(0,len(in_bp)))
for x, y in zip(nin_bp, in_bp):
    plt.text(y, bpcopy[y], str(x), color="red", fontsize=12)

#plt.show()

plt.figure('ECG')
markecg = in_ecg
plt.plot(numbers, ecg, '-gx', markevery=markecg)
nin_ecg = list(range(0,len(in_ecg)))
for x, y in zip(nin_ecg, in_ecg):
    plt.text(y, ecgcopy[y], str(x), color="red", fontsize=12)

#plt.show()


end = time.time()
print("Execution time: ", (end - start))
plt.show()