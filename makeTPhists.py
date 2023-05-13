#!/nfshome0/lumipro/brilconda3/bin/python
#source /cvmfs/cms-bril.cern.ch/brilconda3/bin/activate
#sshfs -o reconnect utca:/scratch/bcm1futca/tempdata/hd5/ ~/private/data_mount
#sshfs -o reconnect utca:/localdata/comissioning22/ ~/private/data_mountbis
#sshfs -o reconnect utca:/localdata/comissioning22_processor ~/private/data_mountris

import tables 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import seaborn as sns
from scipy.stats import norm, mode
from statistics import mean, stdev
from datetime import datetime
#from decimal import Decimal #import statistics #from statistics import mode #import pybaselines #from pybaselines import utils #import peakutils


index_TP = 7420
allChannels = [1,2,7,8,10,13,14,20,27,28,33,37,38,19,34,22,36,46,41,42]
alldataPathAlias = ["../data_mount/f8a7cd2d-*.hd5", "../data_mount/5b0139d5-*.hd5", "../data_mount/29311fe7-*.hd5", "../data_mountbis/97941fbd-*.hd5"]# "../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5"]
means_tot = [] ; sigmas_tot = [] ; names_tot = [] ; temperatures_tot = [] 
modeTP_tot = []



def set_parameters(pathAlias):
    name = ""
    if pathAlias == "../data_mount/f8a7cd2d-*.hd5":
        name= "March_31"
    elif pathAlias == "../data_mount/5b0139d5-*.hd5":
        name= "April_4"
    elif pathAlias == "../data_mount/29311fe7-*.hd5":
        name= "April_6"
    elif pathAlias == "../data_mount/bdd79791-*.hd5":
        name='April_22'
    elif pathAlias == "../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5" or pathAlias == "../data_mountbis/97941fbd-*.hd5":
        name = "June_20"

    temperature = 0
    if pathAlias == "../data_mount/f8a7cd2d-*.hd5":
       temperature = -12.95
    elif pathAlias == "../data_mount/5b0139d5-*.hd5":
        temperature = -13.75
    elif pathAlias == "../data_mount/29311fe7-*.hd5":
        temperature = -14.75
    elif pathAlias == "../data_mount/bdd79791-*.hd5":
        temperature = -16
    elif pathAlias == "../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5" or pathAlias == "../data_mountbis/97941fbd-*.hd5":
        temperature = -9.02
    
    return name, temperature
    
# reading correct raw data table per channel and per file
def readData(channel, fileName):
    data = []
    with tables.open_file(fileName, "r") as h5file:
        if "/bcm1futcarawdata" in h5file:
            for row in h5file.get_node("/bcm1futcarawdata").iterrows():           
                if row['algoid'] == 100 and row['channelid'] == channel: 
                    data.append(row['data'])
                    #print("ciao")
    #print(data)
    return data


#plot the orbits
def plotorbit(channel, countsLimit, pathAlias):
    h5_allNames = glob.glob(pathAlias)
    for h5file_name in h5_allNames[1:]:
       # print(h5file_name, channel)
        if 'allData' in locals(): 
            data = readData(channel, h5file_name)
            if len(data) > 0:
                allData = np.vstack([allData, data])   
        else: allData = np.asarray(readData(channel, h5file_name))

        #print(f"{len(allData)/countsLimit*100}% done")
        if len(allData) > countsLimit:
            break
            
    name, temperature = set_parameters(pathAlias)
   
    ampl = 0
    mean, TPlocation = makeTPampl(channel, countsLimit, pathAlias)
    a = makeBaseline(channel, countsLimit , pathAlias)
    b = appendBaseline(a, pathAlias)
    if pathAlias == "../data_mount/f8a7cd2d-*.hd5":
            c = np.array(b)
            d = c[:,0]
            j = d.tolist()
            print (j)
            ampl = mean + j
    else: 
            c = np.array(b)
            d = c.tolist()
            print (d)
            ampl = mean + d

    if channel == 1 and (pathAlias == "../data_mount/f8a7cd2d-*.hd5" or pathAlias == "../data_mount/5b0139d5-*.hd5"):
        for row in allData:
            plt.plot(row)                           # to see how the raw orbits look like
            plt.plot(TPlocation, ampl, marker="o", markersize= 6, markeredgecolor="red", markerfacecolor="green")
            plt.show()
            #plt.xlim(TPlocation - 60, TPlocation + 60)
            plt.title(f'Orbit-channel %s, Temperature: %.2f $^\circ$ C ' % (channel, temperature))
            plt.savefig(f'orbits/amplt_{name}_channel{channel}.png')
            plt.close()

    elif pathAlias == "../data_mountbis/97941fbd-*.hd5":
        for row in allData:
            plt.plot(row)                           # to see how the raw orbits look like
            plt.plot(TPlocation, ampl, marker="o", markersize= 6, markeredgecolor="red", markerfacecolor="green")
            plt.show()
            #plt.xlim(TPlocation - 60, TPlocation + 60)
            plt.title(f'Orbit-channel %s, Temperature: %.2f $^\circ$ C ' % (channel, temperature))
            plt.savefig(f'orbits/amplt_{name}_channel{channel}.png')
            plt.close()

    else:
        for row in allData:
            plt.plot(row)                           # to see how the raw orbits look like
            plt.plot(TPlocation, ampl, marker="o", markersize= 6, markeredgecolor="red", markerfacecolor="green")
            plt.show()
            plt.xlim(TPlocation - 60, TPlocation + 60)
            plt.title(f'Orbit-channel %s, Temperature: %.2f $^\circ$ C ' % (channel, temperature))
            plt.savefig(f'orbits/amplt_{name}_channel{channel}.png')
            plt.close()

  

   

def simpleorbit (channel, countsLimit, pathAlias):
    # finding Test Pulse based on constant value where it's expected
    h5_allNames = glob.glob(pathAlias)
    for h5file_name in h5_allNames[1:]:
       # print(h5file_name, channel)
        if 'allData' in locals(): 
            data = readData(channel, h5file_name)
            if len(data) > 0:
                allData = np.vstack([allData, data])   
        else: allData = np.asarray(readData(channel, h5file_name))

        #print(f"{len(allData)/countsLimit*100}% done")
        if len(allData) > countsLimit:
            break
    #print(data)


    for row in allData:
        plt.plot(row)							# to see how the raw orbits look like
        plt.show()
        plt.savefig(f"orbit_channel{channel}.png")
        plt.close()
        
    
# finding Test Pulse based on constant value where it's expected
def findTPs(data):
    amplTP = []
    amplLOC = []
    for row in data:
        amplTP.append(max(row[index_TP-200:index_TP+200])-mode(row)[0].squeeze())
        #amplTP.append(max(row)-mode(row)[0].squeeze())

        #plt.plot(row)							# to see how the raw orbits look like
        #plt.show()
        #print(index_TP, max(row), np.argmax(row))  			# to check if TP is not somewhere else than expected
    print (amplTP) 
    return amplTP

# finding Test Pulse based on constant value where it's expected
def findTPslocation(data):
    
    amplTP = []
    amplLOC = []
    for row in data:
        amplTP.append(max(row)-mode(row)[0].squeeze())
        #amplTP.append(max(row[index_TP-200:index_TP+200])-mode(row)[0].squeeze())

        plt.plot(row)							# to see how the raw orbits look like
        plt.show()
        plt.savefig('orbit.png')
        
        amplLOC.append(np.argmax(row))
        print(index_TP, max(row), np.argmax(row))  			# to check if TP is not somewhere else than expected
    return amplTP, amplLOC

#finding the array for the amplitudes
def makeTPampl(channel, countsLimit, pathAlias):

    if pathAlias == "../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5":
        data = []
        with tables.open_file(pathAlias, "r") as h5file:
            if "/bcm1futcarawdata" in h5file:
                for row in h5file.get_node("/bcm1futcarawdata").iterrows():           
                    if row['algoid'] == 100 and row['channelid'] == channel: 
                        data.append(row['data'])
        allData = data               

    else:
        h5_allNames = glob.glob(pathAlias)
        for h5file_name in h5_allNames[1:]:
        # print(h5file_name, channel)
            if 'allData' in locals(): 
                data = readData(channel, h5file_name)
                if len(data) > 0:
                    allData = np.vstack([allData, data])   
            else: allData = np.asarray(readData(channel, h5file_name))

        #print(f"{len(allData)/countsLimit*100}% done")
            if len(allData) > countsLimit:
                break
    if pathAlias == "../data_mountbis/97941fbd-*.hd5":
        amplTP, amplLOC = findTPslocation(allData)
        (mean, sigma) = norm.fit(amplTP)
        mu = amplLOC[-1]
        return mean, mu
    else:
        amplTP, amplLOC = findTPslocation(allData)
        (mean, sigma) = norm.fit(amplTP)
        (mu, sig) = norm.fit(amplLOC)
        return mean, mu
    

#find the baseline value
def findBaseline(data):
    modeTP = []
    for row in data:
        row = np.delete(row,np.argmax(row))	
        modeTP.append(mode(row)[0].squeeze())	
    #print(modeTP) #n, bins, patches = plt.hist(modeTP, len(np.unique(np.array(modeTP))), facecolor='blue', alpha=0.6) #plt.savefig("dummy6_name.png")
    return modeTP
       # plt.savefig("dummy5_name.png") # print(modeTP) # print(np.shape(modeTP)) #(mean, sigma) = norm.fit(modeTP)  #print(mean, sigma) # print(np.shape(allData))
       # for row in allData: # plt.plot(row) # plt.savefig("dummy4_name.png")
       # print(index_TP, max(allData[0]), np.argmax(allData[0])) # to check if TP is not somewhere else than expected # print(np.size(allData[0]))# print(np.shape(allData[0]))  #print(max(allData[0])) # y=np.delete(allData[0],np.argmax((allData[0])))# print(max(y))#print(np.shape(y))  # print(mode(y)[0]) #plt.plot(y)#plt.savefig("dummy3_name.png") #x=np.arange(0,106919)  #sns.regplot(x=x,y=y)  #plt.savefig("output.png")

#return the baseline value
def makeBaseline(channel, countsLimit, pathAlias):
    if pathAlias == "../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5":
        data = []
        with tables.open_file(pathAlias, "r") as h5file:
            if "/bcm1futcarawdata" in h5file:
                for row in h5file.get_node("/bcm1futcarawdata").iterrows():           
                    if row['algoid'] == 100 and row['channelid'] == channel: 
                        data.append(row['data'])
        allData = data               

    else:
        h5_allNames = glob.glob(pathAlias)
        for h5file_name in h5_allNames[1:]:
            # print(h5file_name, channel)
            if 'allData' in locals(): 
                data = readData(channel, h5file_name)
                if len(data) > 0:
                    allData = np.vstack([allData, data])   
            else: allData = np.asarray(readData(channel, h5file_name))

        #print(f"{len(allData)/countsLimit*100}% done")
            if len(allData) > countsLimit:
                break

    modeTP = findBaseline(allData)
    print(modeTP)
    print(np.shape(modeTP))
    return(modeTP)
    #n, bins, patches = plt.hist(modeTP, len(np.unique(np.array(modeTP))), facecolor='blue', alpha=0.6)
    #plt.ylim(0,2)
    #plt.savefig("dummy7_name.pdf")


#return the baseline values histogram
def histBaseline(channel, countsLimit, pathAlias):

    name, temperature = set_parameters(pathAlias)


    if pathAlias == "../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5":
        data = []
        with tables.open_file(pathAlias, "r") as h5file:
            if "/bcm1futcarawdata" in h5file:
                for row in h5file.get_node("/bcm1futcarawdata").iterrows():           
                    if row['algoid'] == 100 and row['channelid'] == channel: 
                        data.append(row['data'])
        allData = data               

    else:
        h5_allNames = glob.glob(pathAlias)
        for h5file_name in h5_allNames[1:]:
            # print(h5file_name, channel)
            if 'allData' in locals(): 
                data = readData(channel, h5file_name)
                if len(data) > 0:
                    allData = np.vstack([allData, data])   
            else: allData = np.asarray(readData(channel, h5file_name))

        #print(f"{len(allData)/countsLimit*100}% done")
            if len(allData) > countsLimit:
                break

    modeTP = findBaseline(allData)
    print(modeTP)
    print(np.shape(modeTP))

    num_bins = 7
    n, bins, patches = plt.hist(modeTP, num_bins, facecolor='blue', alpha=0.6)
    plt.xlabel('Baseline')
    plt.ylabel('Count') # norm.
    plt.title(r'$\mathrm{Channel\ %d, TP\ Histogram}$' % (channel))
    plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C' % (name, temperature))
    plt.grid(True)
    #plt.savefig(f'histograms/counts_{countsLimit}date_{name}_channel{channel}.png') 
    plt.savefig("dummy.png")
    plt.close()
    #plt.ylim(0,2)
    #plt.savefig("dummy7_name.pdf")

#append the baseline 
def appendBaseline(modeTP, path):
    if path == "../data_mount/f8a7cd2d-*.hd5":
        modeTP_tot.append(modeTP)
        return modeTP_tot
    elif path == "../data_mount/5b0139d5-*.hd5":
        modeTP_tot.append(modeTP[0])
        return modeTP_tot
    elif path == "../data_mount/29311fe7-*.hd5":
        modeTP_tot.append(modeTP[0])
        return modeTP_tot
    elif path == "../data_mount/bdd79791-*.hd5":
        modeTP_tot.append(modeTP[0])
        return modeTP_tot
    elif path =="../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5" or path == "../data_mountbis/97941fbd-*.hd5":
        modeTP_tot.append(modeTP[0])
        #print(modeTP_tot)
        return modeTP_tot
    
#plot the baseline distribution
def makeTPmodehist(countsLimit, path):
    name, temperature = set_parameters(path)

    for channel in allChannels:
        a = makeBaseline(channel, countsLimit , path)
        b = appendBaseline(a, path)
    print(np.shape(b))
    print (b)
    
    if path == "../data_mount/f8a7cd2d-*.hd5":
            c = np.array(b)
            print(c)
            print(np.shape(c))
            d = c[:,0]
            print (d)
            j = d.tolist()
            print (j)
            (mu, sigma) = norm.fit(j)
            print(mu, sigma)
            n, bins, patches = plt.hist(j, len(np.unique(np.array(j))), facecolor='red', alpha=0.6)
            plt.xlabel('Baseline')
            plt.ylabel('Count') # norm.
            plt.title('Baseline histogram')
            # add a 'best fit' line
            #y = norm.pdf(bins, mu, sigma)
            #l = plt.plot(bins, y, 'b--', linewidth=2)
            plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (name, temperature))
            plt.grid(True)
            plt.savefig(f"Baseline_Date{name}_Temp{temperature}.pdf") 
            plt.close()
            
    else: 
            c = np.array(b)
            print(c)
            print(np.shape(c))
            d = c.tolist()
            print (d)
            (mu, sigma) = norm.fit(d)
            print(mu, sigma)
            n, bins, patches = plt.hist(d, len(np.unique(np.array(d))), facecolor='red', alpha=0.6) #density = True
            plt.xlabel('Baseline')
            plt.ylabel('Count') # norm.
            plt.title('Baseline histogram')
            # add a 'best fit' line
            #y = norm.pdf(bins, mu, sigma)
            #l = plt.plot(bins, y, 'b--', linewidth=2)
            plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (name, temperature))
            plt.grid(True)
            plt.savefig(f"Baseline_Date{name}_Temp{temperature}.png") 
            plt.close()
             
#plot the baseline scatter
def makeTPmodescatter(countsLimit, path):
    name, temperature = set_parameters(path)

    for channel in allChannels:
        a = makeBaseline(channel, countsLimit , path)
        b = appendBaseline(a, path)
    print(np.shape(b))
    print (b)
    
    if path == "../data_mount/f8a7cd2d-*.hd5":
            c = np.array(b)
            d = c[:,0]
            j = d.tolist()
            fig = plt.scatter(allChannels, j)
            plt.title('Baseline plot')
            plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (name, temperature))
            plt.xlabel('Channel')
            plt.ylabel('Baseline value')
            plt.xticks( allChannels, allChannels, rotation=60)
            plt.yticks( range(121,133,1) )
            plt.grid(True)
            plt.savefig(f"PlotBaseline_Date{name}_Temp{temperature}.pdf") 
            plt.close()
           
    else: 
            c = np.array(b)
            d = c.tolist()
            fig = plt.scatter(allChannels, d)
            plt.title('Baseline plot')
            plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (name, temperature))
            plt.xlabel('Channel')
            plt.ylabel('Baseline value')
            #plt.xticks( range(0, 20, 1), allChannels)
            plt.xticks (allChannels, allChannels, rotation=60)
            plt.yticks( range(121,133,1) )
            plt.grid(True)
            plt.savefig(f"PlotBaseline_Date{name}_Temp{temperature}.png") 
            plt.close()

#plot the baseline scatter
def makeTPmodeerrorscatter(countsLimit, path):
    name, temperature = set_parameters(path)


    for channel in allChannels:
        a = makeBaseline(channel, countsLimit , path)
        b = appendBaseline(a, path)

    e,f,g,h = makeTPerrorMODE(allChannels, 1, path)
    h,i,l,m = total_append(e,f,g,h, allChannels)

    if path == "../data_mount/f8a7cd2d-*.hd5":
            c = np.array(b)
            d = c[:,0]
            j = d.tolist()
            fig = plt.scatter(allChannels,j)
            plt.errorbar(allChannels,j,yerr=i[0], ecolor='blue', markerfacecolor= 'blue', fmt='o',capsize=4, label=f' %.2f $^\circ$ C ' % (m[0]))
            plt.title('Baseline plot')
            plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (name, temperature))
            plt.xlabel('Channel')
            plt.ylabel('Baseline value')
            plt.xticks( allChannels, allChannels, rotation=60)
            #plt.yticks( range(121,133,1) )
            plt.yticks( range(118,136,1) )
            plt.grid(True)
            plt.savefig(f"PpplotBaseline_Date{name}_Temp{temperature}.png") 
            plt.close()

    else: 
            c = np.array(b)
            d = c.tolist()
            fig = plt.scatter(allChannels, d)
            plt.errorbar(allChannels,d,yerr=i[0], ecolor='blue', markerfacecolor= 'blue', fmt='o',capsize=4, label=f' %.2f $^\circ$ C ' % (m[0]))
            plt.title('Baseline plot')
            plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (name, temperature))
            plt.xlabel('Channel')
            plt.ylabel('Baseline value')
            #plt.xticks( range(0, 20, 1), allChannels)
            plt.xticks (allChannels, allChannels, rotation=60)
            plt.yticks( range(118,136,1) )
            #plt.yticks( range(118,134,1) )
            plt.grid(True)
            plt.savefig(f"PpplotBaseline_Date{name}_Temp{temperature}.png") 
            plt.close()

def makeTPmodesummary():
    
    for path in alldataPathAlias:
        e,f,g,h = makeTPerrorMODE(allChannels, 1, path)
        a,b,c,d = total_append(e,f,g,h, allChannels)

    s = np.array(
        [[130, 127, 125, 126, 126, 127, 127, 127, 126, 129, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127],
        [130, 127, 125, 126, 126, 127, 127, 127, 126, 130, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127], 
        [130, 127, 125, 126, 126, 127, 127, 127, 126, 130, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127],
        [130, 127, 125, 126, 126, 127, 127, 127, 126, 129, 127, 129, 126, 126, 128, 128, 131, 121, 131, 127]]
        )

    k = s.tolist()

   
        
    fig = plt.scatter(allChannels, k[0])
    plt.scatter(allChannels, k[1], color='red')
    plt.scatter(allChannels, k[2], color='green')
    plt.errorbar(allChannels,k[0],yerr=b[0], ecolor='blue', markerfacecolor= 'blue', fmt='o',capsize=4, label=f' %.2f $^\circ$ C ' % (d[0]))
    plt.errorbar(allChannels,k[1],yerr=b[1], ecolor='red', markerfacecolor= 'red', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[1]))
    plt.errorbar(allChannels,k[2],yerr=b[2], ecolor='green',markerfacecolor= 'green', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[2]))
    plt.errorbar(allChannels,k[3],yerr=b[2], ecolor='magenta',markerfacecolor= 'magenta', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[3]))
    plt.xlabel('Channel')
    plt.ylabel('Baseline')
    plt.title('Baseline summary')
    plt.xticks(allChannels, allChannels, rotation=60)
    plt.yticks( range(118,136,1) )
    plt.grid(True)
    plt.legend()
    plt.savefig(f"XlotBaseline1.png")
    plt.close()
    
    
    #

def makeTP3modesummary():
    for path in alldataPathAlias:
        e,f,g,h = makeTPerrorMODE(allChannels, 1, path)
        a,b,c,d = total_append(e,f,g,h, allChannels)

    t = np.array([130, 127, 125, 126, 126, 127, 127, 127, 126, 129, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127])
    w = np.array([130, 127, 125, 126, 126, 127, 127, 127, 126, 130, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127])
    s = np.array([130, 127, 125, 126, 126, 127, 127, 127, 126, 130, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127])
    h = np.array ([130, 127, 125, 126, 126, 127, 127, 127, 126, 129, 127, 129, 126, 126, 128, 128, 131, 121, 131, 127])

    x = t.tolist()
    y = w.tolist()
    k = s.tolist()
    j = h.tolist()

    (mean1, sigma1) = norm.fit(t)
    (mean2, sigma2) = norm.fit(w)
    (mean3, sigma3) = norm.fit(s)
    (mean4, sigma4) = norm.fit (h)
            
    fig = plt.scatter(allChannels, x)
    plt.errorbar(allChannels,x,yerr=sigma1, ecolor='blue', fmt='o',capsize=4, label=f' %.2f $^\circ$ C ' % (d[0]))
    plt.xlabel('Channel')
    plt.ylabel('Baseline')
    plt.title('Baseline summary')
    plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (c[0], d[0]))
    plt.xticks(allChannels, allChannels, rotation=60)
    #plt.yticks( range(118,136,1) )
    plt.grid(True)
    plt.legend()
    plt.savefig(f"xlotBaseline.png")
    plt.close()

    fig2 = plt.scatter(allChannels, y)
    plt.errorbar(allChannels,y,yerr=sigma2, fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[1]))
    plt.xlabel('Channel')
    plt.ylabel('Baseline')
    plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (c[1], d[1]))
    plt.title('Baseline summary')
    plt.xticks(allChannels, allChannels, rotation=60)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ylotBaseline.png")
    plt.close()

    fig3 = plt.scatter(allChannels, k)
    plt.errorbar(allChannels,k,yerr=sigma3, fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[2]))
    plt.xlabel('Channel')
    plt.ylabel('Baseline')
    plt.title('Baseline summary')
    plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (c[2], d[2]))
    plt.xticks(allChannels, allChannels, rotation=60)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"tlotBaseline.png")
    plt.close()

    fig4 = plt.scatter(allChannels, j)
    plt.errorbar(allChannels,j,yerr=sigma3, fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[3]))
    plt.xlabel('Channel')
    plt.ylabel('Baseline')
    plt.title('Baseline summary')
    plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (c[3], d[3]))
    plt.xticks(allChannels, allChannels, rotation=60)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"jlotBaseline.png")
    plt.close()

def makeTP2modesummary():

    for path in alldataPathAlias:
        e,f,g,h = makeTPerrorMODE(allChannels, 1000, path)
        a,b,c,d = total_append(e,f,g,h, allChannels)

    t = np.array([130, 127, 125, 126, 126, 127, 127, 127, 126, 129, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127])
    w = np.array([130, 127, 125, 126, 126, 127, 127, 127, 126, 130, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127])
    s = np.array([130, 127, 125, 126, 126, 127, 127, 127, 126, 130, 127, 129, 126, 126, 128, 128, 131, 122, 131, 127])
    h = np.array ([130, 127, 125, 126, 126, 127, 127, 127, 126, 129, 127, 129, 126, 126, 128, 128, 131, 121, 131, 127])

    x = t.tolist()
    y = w.tolist()
    k = s.tolist()
    j = h.tolist()

    (mean1, sigma1) = norm.fit(t)
    (mean2, sigma2) = norm.fit(w)
    (mean3, sigma3) = norm.fit(s)
    (mean4, sigma4) = norm.fit(j)
    
    fig = plt.scatter(allChannels, x)
    plt.scatter(allChannels, y, color='red')
    plt.scatter(allChannels, k, color='green')
    plt.scatter(allChannels, j, color='yellow')
    plt.errorbar(allChannels,x,yerr=sigma1, ecolor='blue', markerfacecolor= 'blue', fmt='o',capsize=4, label=f' %.2f $^\circ$ C ' % (d[0]))
    plt.errorbar(allChannels,y,yerr=sigma2, ecolor='red', markerfacecolor= 'red', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[1]))
    plt.errorbar(allChannels,k,yerr=sigma3, ecolor='green',markerfacecolor= 'green', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[2]))
    plt.errorbar(allChannels,j,yerr=sigma3, ecolor='yellow',markerfacecolor= 'yellow', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[3]))
    plt.xlabel('Channel')
    plt.ylabel('Baseline')
    plt.title('Baseline summary')
    plt.xticks(allChannels, allChannels, rotation=60)
    #plt.yticks( range(118,136,1) )
    plt.grid(True)
    plt.legend()
    plt.savefig(f"OlotBaseline1000.png")
    plt.close()

    
#find the baseline for the amplitude histograms
def modeTPs(amplTP):
    return mode(amplTP)[0]
    print(amplsTP)
    
# plot Test Pulse amplitude histograms per channel, countsLimit sets number of pulses included in the histogram
def makeTPhists(channel, countsLimit, pathAlias):
    
    name, temperature = set_parameters(pathAlias)



    if pathAlias == "../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5":
        data = []
        with tables.open_file(pathAlias, "r") as h5file:
            if "/bcm1futcarawdata" in h5file:
                for row in h5file.get_node("/bcm1futcarawdata").iterrows():           
                    if row['algoid'] == 100 and row['channelid'] == channel: 
                        data.append(row['data'])
        allData = data               

    else:
        h5_allNames = glob.glob(pathAlias)
        for h5file_name in h5_allNames[1:]:
            # print(h5file_name, channel)
            if 'allData' in locals(): 
                data = readData(channel, h5file_name)
                if len(data) > 0:
                    allData = np.vstack([allData, data])   
            else: allData = np.asarray(readData(channel, h5file_name))

        #print(f"{len(allData)/countsLimit*100}% done")
            if len(allData) > countsLimit:
                break

    amplsTP = findTPs(allData)
    (mean, sigma) = norm.fit(amplsTP)

    modeTP=modeTPs(amplsTP)


   

    n, bins, patches = plt.hist(amplsTP, len(np.unique(np.array(amplsTP))), facecolor='blue', alpha=0.6) # , density=True
   # plot with date, temperature info
    with PdfPages(f'TPhists/channel{channel}_TPhist{countsLimit}_Date{name}_Temp{temperature}.pdf') as pdf:
        #plt.axvline(modeTP, color='red', label='Mode')
        plt.xlabel('Amplitude')
        plt.ylabel('Count') # norm.
        plt.title(r'$\mathrm{Channel\ %d, TP\ Histogram:}\ \mu=%.2f,\ \sigma=%.2f$' % (channel, mean, sigma))
        plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C' % (name, temperature))
        plt.grid(True)
        pdf.savefig()
        plt.close()
        #plt.yticks([minim],['Offset'])
        #\mathrm{date\ %d}
    
# plot overview of the mean test pulse for range of channels (list as an input), countsLimit sets number of pulses to be considered per channel 
def makeTPsummary(channels, countsLimit, pathAlias):

    name, temperature = set_parameters(pathAlias)
    
    
    means = []; sigmas = []
    
    h5_allNames = glob.glob(pathAlias)
    print(len(h5_allNames), h5_allNames)
    means = []; sigmas = []
    for channel in channels:
        print(channel)
        for h5file_name in h5_allNames:
            print(h5file_name)
            if 'allData' in locals(): 
                data = readData(channel, h5file_name)
                if len(data) > 0:
                    allData = np.vstack([allData, data])   
            else: allData = np.asarray(readData(channel, h5file_name))

            if len(allData) > countsLimit:
                 break

    amplsTP = findTPs(allData)
    del allData
    (mean, sigma) = norm.fit(amplsTP)
    means.append(mean)
    sigmas.append(sigma)
    
    with PdfPages(f'TPhists/TPampl_summary{countsLimit}_Date{name}_Temp{temperature}.pdf') as pdf:
        plt.xlabel('Channel')
        plt.ylabel('Amplitude')
        plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (name, temperature))
        plt.errorbar(channels,means,yerr=sigmas, fmt='o',capsize=4)
        plt.xticks(channels, rotation=60)
        plt.grid(True)
        pdf.savefig()
        plt.close()
  
# plot overview of the mean test pulse for range of channels (list as an input), countsLimit sets number of pulses to be considered per channel 
def makeTPerror(channels, countsLimit, pathAlias):

    name, temperature = set_parameters(pathAlias)

    h5_allNames = glob.glob(pathAlias)
    print(len(h5_allNames), h5_allNames)
    means = []; sigmas = []
    for channel in channels:
        print(channel)
        for h5file_name in h5_allNames:
            print(h5file_name)
            if 'allData' in locals(): 
                data = readData(channel, h5file_name)
                if len(data) > 0:
                    allData = np.vstack([allData, data])   
            else: allData = np.asarray(readData(channel, h5file_name))

            if len(allData) > countsLimit:
                 break

        amplsTP = findTPs(allData)
        del allData
        (mean, sigma) = norm.fit(amplsTP)
        means.append(mean)
        sigmas.append(sigma)
    
    return means, sigmas, name, temperature

def makeTPerrorMODE(channels, countsLimit, pathAlias):

    

    name, temperature = set_parameters(pathAlias)

    h5_allNames = glob.glob(pathAlias)
    print(len(h5_allNames), h5_allNames)
    means = []; sigmas = []
    for channel in channels:
        print(channel)
        for h5file_name in h5_allNames:
            print(h5file_name)
            if 'allData' in locals(): 
                data = readData(channel, h5file_name)
                if len(data) > 0:
                    allData = np.vstack([allData, data])   
            else: allData = np.asarray(readData(channel, h5file_name))

            if len(allData) > countsLimit:
                 break

        modesTP = findBaseline(allData)
        del allData
        (mean, sigma) = norm.fit(modesTP)
        means.append(mean)
        sigmas.append(sigma)
    
    return means, sigmas, name, temperature
#append to generate total array for errorbar plot
def total_append(mean, sigma, name, temperature, channels):
    means_tot.append(mean)
    sigmas_tot.append(sigma)
    names_tot.append(name)
    temperatures_tot.append(temperature)

    return means_tot, sigmas_tot, names_tot, temperatures_tot

#plot errorbar for a single temperature  
def makeTPcomparison(mean, sigma, name, temperature):
    with PdfPages(f'TPhists/TPampl_summary_Date{name}_Temp{temperature}.pdf') as pdf:
        plt.xlabel('Channel')
        plt.ylabel('Amplitude')
        plt.suptitle(f'Date: %s, Temperature: %.2f $^\circ$ C ' % (name[0], temperature[0]))
        plt.errorbar(allChannels,mean[0],yerr=sigma[0], fmt='o',capsize=4)
        plt.xticks(allChannels, rotation=60)
        plt.grid(True)
        pdf.savefig()
        plt.close()


#plot the errorbar together for the different temperatures
def makeTPamplsummary():

    for path in alldataPathAlias:
        e,f,g,h = makeTPerror(allChannels, 1, path)
        a,b,c,d = total_append(e,f,g,h, allChannels)
        
    
    fig = plt.errorbar(allChannels,a[0],yerr=b[0], ecolor='blue', markerfacecolor= 'blue', fmt='o',capsize=4, label=f' %.2f $^\circ$ C ' % (d[0]))
    plt.errorbar(allChannels,a[1],yerr=b[1], ecolor='red', markerfacecolor= 'red', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[1]))
    plt.errorbar(allChannels,a[2],yerr=b[2], ecolor='green',markerfacecolor= 'green', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[2]))
    plt.errorbar(allChannels,a[2],yerr=b[2], ecolor='black',markerfacecolor= 'black', fmt='o',capsize=4, label=f'%.2f $^\circ$ C ' % (d[3])) 
    plt.xlabel('Channel')
    plt.ylabel('Amplitude')
    plt.title('Amplitude summary')
    #plt.suptitle(f'Date: %s, Temperature: %f $^\circ$ C ' % (name[0], temperature[0]))
    plt.xticks(allChannels, rotation=60)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(f'TPampl_summary.png')
    plt.close()



# plot Test Pulse mean amplitude evolution for list of channels and given pulse count limit
def makeTPevo(channels, countsLimit, pathAlias):

    h5_allNames = glob.glob(pathAlias)
    with PdfPages(f'TPhists/TP_evo_allChannels_count{countsLimit}.pdf') as pdf:
        for channel in channels:
            print(channel)
            timeSteps = []; avAmpl = []; stdAmpl = []
            for h5file_name in h5_allNames:
                print(h5file_name )
                if 'allData' in locals(): 
                    data = readData(channel, h5file_name)
                    if len(data) > 0:
                        allData = np.vstack([allData, data])   
                else: allData = np.asarray(readData(channel, h5file_name))

                if len(avAmpl) > countsLimit:
                    break

                dateStr = h5file_name.split('_')[1]
                timeSteps.append(datetime(int(dateStr[0:2])+2000, int(dateStr[2:4]), int(dateStr[4:6]), int(dateStr[6:8]), int(dateStr[8:10])))
                amplsTP = findTPs(allData)
                avAmpl.append(np.mean(amplsTP))
                stdAmpl.append(np.std(amplsTP))
                del allData
                print(f"{len(avAmpl)/countsLimit*100}% done")

            plt.errorbar(timeSteps,avAmpl,yerr=stdAmpl, fmt='o',capsize=4)
            plt.xlabel('Time')
            plt.ylabel('Amplitude') 
            plt.title(r'$\mathrm{Channel}\ %d$' % (channel))
            plt.grid(True)
            plt.xticks(rotation=20)
            plt.show()
            pdf.savefig()
            plt.close()

# example calling the plotting functions, dataPathAlias gives string to look for in filenames - can be adjusted per function

dataPathAlias = "../data_mount/f8a7cd2d-*.hd5" #March31-12.95deg
dataPathAlias2 = "../data_mount/5b0139d5-*.hd5" #April4-13.75deg
dataPathAlias3 = "../data_mount/29311fe7-*.hd5"  #April6-14.75deg
#dataPathAlias4 = "../data_mount/bdd79791-*.hd5" #April22
dataPathAlias5="../data_mount/764452d9-*.hd5" #March31-12.95deg 
dataPathAlias6 = "../data_mountbis/97941fbd-dd0a-4326-86ba-fe8c731e7f6e_2206231515_0.hd5" #June20-9.02deg
dataPathAlias7 = "../data_mountbis/97941fbd-*.hd5" #June20-9.02deg

