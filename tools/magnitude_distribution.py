import sys, os, urllib, urlparse, pyfits, pickle, optparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from classify import predict
from sklearn import preprocessing

CANDIDATE_URL = "http://psweb.mp.qub.ac.uk/sne/ps13pi/psdb/candidate/"
SERVERNAME = "psweb.mp.qub.ac.uk/sne/ps13pi/psdb/lightcurve/"
MAGNITUDE_REPOSITORY = "/Users/dew/development/PS1-Real-Bogus/data/3pi/mags/"
IMAGE_REPOSITORY = "/Users/dew/myscripts/machine_learning/data/3pi/detectionlist/"
OUT_FILE = "3pi_realObject_data.csv"

def dataDownload(remotedir, filename, localdir="./"):
    remoteaddr = 'http://%s%s' % (remotedir, filename)
    #print remoteaddr
    (scheme, server, path, params, query, frag) = urlparse.urlparse(remoteaddr)
    localname = os.path.split(path)[1]
    #print remoteaddr, localname                                                                     
    try:
        # retrieve remoteaddr from server and store in localname on client                           
        urllib.urlretrieve(remoteaddr, localdir+localname+".txt")
    except IOError, e:
        print "ERROR: Failed to download. Error is: %s"% e.errno
        
def checkWebInfo(id, image_id, mjd):
    url = CANDIDATE_URL + id + "/"
    servername = SERVERNAME
    html = urllib.urlopen(url).read()
    index = html.index("Object List")
    objectList = html[index:index+50]
    index = html.index("Spectral Type:")
    type = html[index+22:index+30].strip("</h3><").strip("</h3")
    index = html.index("PS1 Name:")
    name = html[index+17:index+26].strip("<").strip("</")
    try:
        index = html.index("<td>"+image_id)
        mag = html[index-80:index-70].strip(" <td>").strip("</")
        assert (float(mag) < 25.0 and float(mag) > 0),"here"
    except Exception, e:
        print e
        filename = id+".txt"
        try:
            open(MAGNITUDE_REPOSITORY+filename, "r")
        except:
            dataDownload(servername, id, MAGNITUDE_REPOSITORY)
        for line in open(MAGNITUDE_REPOSITORY+filename, "r").readlines():
            if "#" in line or line.rstrip() == "":
                continue
            print line.strip()
            if (float(line.rstrip().split(" ")[0][:9]) - mjd) < 1e-7:
                if line.rstrip().split(" ")[1] == "None":
                    continue
                mag = float(float(line.rstrip().split(" ")[1]))
            else:
                mag = 0
    if type == "d":
        index = html.index("Contextual Classification:")
        type = "("+html[index+33:index+50].strip("</h3></div>\n").strip(">")+")"
    if "good" in objectList:
        print "good", type, name, mag
        return "good", type, name, mag
    elif "garbage" in objectList:
        print "garbage", type, name, mag
        return "garbage", type, name, mag
    elif "confirmed" in objectList:
        print "confirmed", type, name, mag
        return "confirmed", type, name, mag
    elif "possible" in objectList:
        print "possible", type, name, mag
        return "possible", type, name, mag
    elif "attic" in objectList:
        print "attic", type, name, mag
        return "attic", type, name, mag
                
def getLightcurveFiles(fileList, clfFile, X):

    clf = pickle.load(open(clfFile, "rb"))
    pred = clf.predict_proba(X)[:,1]
    mags = []
    counter = 1
    output = open(OUT_FILE, "w")
    for i in range (len(fileList)):
        image = fileList[i]
        print image
        id = image.split("_")[0]
        try:
            hdulist = pyfits.open(IMAGE_REPOSITORY + "2/" + image)
            imagefile = IMAGE_REPOSITORY + "2/" + image
        except:
            try:
                hdulist = pyfits.open(IMAGE_REPOSITORY + "1/" + image)
                imagefile = IMAGE_REPOSITORY + "1/" + image
            except:
                hdulist = pyfits.open(IMAGE_REPOSITORY + "0/" + image)
                imagefile = IMAGE_REPOSITORY + "0/" + image                                                   
        header = hdulist[1].header
                                          
        image_id = image.split("_diff.fits")[0]
        mjd = hdulist[1].header["MJD-OBS"]
        objectList, type, name, mag = checkWebInfo(id, image_id, mjd)
        try:                                                      
            mag = float(mag)
        except Exception, e:
            print e
            print id
            print image_id
            print mag
        if "PS1" not in name or name == "h3></di":
            name = id
        counter+=1
        mags.append(mag)
        print name, mag
        output.write(name + "," + str(mjd) + "," + type + "," + \
                     header["HIERARCH FPA.FILTERID"].split(".")[0] + "," + \
                     str(mag) + "," + objectList + "," + \
                     str(pred[i]) + "," + imagefile + "\n")
    output.close()

    plt.hist(mags, bins=50)
    plt.show()
    
def buildDataset(clfFile, dataFile):

    data = sio.loadmat(dataFile)
    fileList = data["images"]
    X = data["X"]
    getLightcurveFiles(fileList, clfFile, X)

def plot_dist(dataset1, dataset2=None, alpha1=1, alpha2=1):

    dataset1_mags = []
    for line in open(dataset1,"r").readlines():
        dataset1_mags.append(float(line.rstrip().split(",")[4]))

    normed = False
    bins = np.arange(0,26,1)

    fig = plt.figure()
    
    font = {"size"   : 30}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=26)

    ax = fig.add_subplot(111)

    ax.axvspan(12.0, 16.0, facecolor="#3366FF", alpha=0.25)

    n1, bins1, patches1 = ax.hist(dataset1_mags, bins=bins, normed=normed, \
                                  color="#66FF33", label="training set", alpha=alpha1)
    ax.plot(bins, np.zeros(np.shape(bins)), "k-")
    ax.set_ylim(ymin=-20)
    ax.set_xlim(xmin=12.9, xmax=24.1)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Frequency")
 
    mid_points = []
    for i in range(len(n1)):
        mid_points.append((bins[i]+bins[i+1]) / 2.0)
        
    for i in range(len(n1)):
        if n1[i] == 0:
             continue
        ax.text(mid_points[i]-0.4, n1[i]+10, str(n1[i]), \
                weight="heavy", size=26, color="k")

    if dataset2 != None:
    
        dataset2_mags = []
        
        for line in open(dataset2,"r").readlines():
            dataset2_mags.append(float(line.rstrip().split(",")[4]))
            
        n2, bins2, patches2 = ax.hist(dataset2_mags, bins=bins, normed=normed, \
                                      color="#FF0066", label="test set", alpha=alpha2)
        fractions = []
        for i in range(len(n1)):
            fractions.append(n2[i]/float(n1[i]))

        not_zero = np.where(np.nan_to_num(fractions) != 0)[0]
        

        mid_points = np.array(mid_points)
        ax2 = ax.twinx()
        ax2.plot(mid_points[not_zero], np.nan_to_num(fractions)[not_zero], zorder=0, lw=3, color="k")
        ax2.plot(mid_points[not_zero], np.nan_to_num(fractions)[not_zero], zorder=0, lw=2, color="#3366FF")
        ax2.scatter(mid_points[not_zero], np.nan_to_num(fractions)[not_zero], zorder=1, lw=2, color= "#3366FF")
        ax2.set_ylim(ymin=-0.005, ymax=np.max(np.nan_to_num(fractions))+0.01*np.max(np.nan_to_num(fractions)))
        ax2.set_xlim(xmin=12.9, xmax=24.1)
        ax2.grid()
        ax2.set_ylabel("Fraction of training set")
    ax.legend(bbox_to_anchor=(0.6, 0.97), borderaxespad=0.)
    plt.show()

def plot_MDR_vs_mag(clfFile, X, fileList, infoFile, threshold=0.5, color="#FF0066"):
    
    pred = predict(clfFile, X)

    mags = []
    for line in open(infoFile,"r").readlines():
        #print line.rstrip().split(",")[-1].split("/")[-1] in set(fileList)
        if line.rstrip().split(",")[-1].split("/")[-1] in set(fileList):
            #print float(line.rstrip().split(",")[4])
            mags.append(float(line.rstrip().split(",")[4]))
 
    font = {"size"   : 20}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=18)
 
    bins = np.arange(14,23,1)
    n, bins, patches = plt.hist(mags, bins=bins)
    #print bins
    bin_allocations = np.digitize(mags, bins)
    #print bin_allocations
    MDRs = []
    for i in range(1,len(bins)):
        if n[i-1] == 0:
            MDRs.append(0)
            continue
        preds_for_bin = pred[np.squeeze(np.where(bin_allocations == i))]

        MDRs.append(np.shape(np.where(preds_for_bin >= threshold))[1] / float(n[i-1]))
    print MDRs
    mid_points = []
    for i in range(len(bins)-1):
        mid_points.append(np.mean([bins[i], bins[i+1]]))

    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    bins = np.arange(14,23,1)
    ax1 = ax2.twinx()
    ax2.set_xlabel("Magnitude")
    ax1.set_ylabel("Frequency")
    ax1.set_ylim(ymin=0-0.01*np.max(n), ymax=np.max(n)+0.01*np.max(n))
    n, bins, patches = ax1.hist(mags, bins=bins, color=color, \
                                alpha=0.25, edgecolor="none")#FF0066
    ax2.set_zorder(ax1.get_zorder()+1)
    ax2.patch.set_visible(False)
    ax2.set_ylim(ymin=-0.01, ymax=1.01)

    #print np.shape(MDRs)
    #print np.shape(mid_points)
    #oldMDRs = [1.0, 0.5555555555555556, 0.6666666666666666, 1.0, 0.6666666666666666, \
    #        0.6097560975609756, 0.2923076923076923, 0.06217616580310881, 0.05673758865248227, \
    #        0.06451612903225806, 0.1794871794871795]

    #ax2.plot(mid_points, oldMDRs, "-",label="old", color = "#66FF33", lw=2, alpha=0.5)
    #ax2.plot(mid_points, oldMDRs, ".", color = "#66FF33", ms=10, markeredgecolor='none', alpha=0.5)

    ax2.plot(mid_points, MDRs, "-",color = "k", lw=3)
    ax2.plot(mid_points, MDRs, "-",label="new", color = color, lw=2)
    ax2.plot(mid_points, MDRs, "o", color = color, ms=5)#3366FF
    ax2.plot(mid_points+[12,18,25], 0.211*np.ones(np.shape(mid_points+[12, 18, 25])), "--", color="k", lw=2)

    ax2.set_ylabel("Missed Detection Rate")
    ax2.set_xlim(xmin=13.9, xmax=22.1)
    ax2.grid()
    ax2.text(14.1,0.215,"0.211", size=18, color="k")


    #ax2.legend()
    plt.show()

def main():
    
    parser = optparse.OptionParser("[!] usage: python magnitude_distribution.py\n"+\
                                   " -F <data file>\n"+\
                                   " -c <classifier file>\n"+\
                                   " -t <threshold [default=0.5]>\n"+\
                                   " -s <data set>\n"+\
                                   " -i <info. file>\n"
                                   " -P <pooled features file [optional]>")
        
    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to analyse")
    parser.add_option("-c", dest="classifierFile", type="string", \
                      help="specify classifier to use")
    parser.add_option("-t", dest="threshold", type="float", \
                      help="specify decision boundary threshold [default=0.5]")
    parser.add_option("-i", dest="infoFile", type="string", \
                      help="specify info file")
    parser.add_option("-s", dest="dataSet", type="string", \
                      help="specify data set to analyse [default=test]")
    parser.add_option("-P", dest="poolFile", type="string", \
                      help="specify pooled features file [optional]")
                                   
    (options, args) = parser.parse_args()
    dataFile = options.dataFile
    classifierFile = options.classifierFile
    threshold = options.threshold
    infoFile = options.infoFile
    dataSet = options.dataSet
    poolFile = options.poolFile

    if dataFile == None or classifierFile == None or infoFile == None:
        print parser.usage
        exit(0)

    if threshold == None:
        threshold = 0.5
    
    if dataSet == None:
        dataSet = "test"

    data = sio.loadmat(dataFile)
    print data.keys()
    if dataSet == "test":
        try:
            X = data["testX"]
            y = np.squeeze(data["testy"])
            files = data["test_files"]
        except KeyError:
            if plot:
                y = np.zeros((np.shape(X)[0],))
            else:
                print "[!] Exiting: Could not load test set from %s" % dataFile
                exit(0)
    elif dataSet == "training":
        try:
            X = data["X"]
            try:
                y = np.squeeze(data["y"])
            except KeyError:
                if fom:
                    print "[!] Exiting: Could not load labels from %s" % dataFile
                    print "[*] FoM calculation is not possible without labels."
                    exit(0)
                else:
                    y = np.zeros((np.shape(X)[0],))
            files = data["images"]
        except KeyError:
            try:
                files = data["train_files"]
            except KeyError, e:
                print e
                try:
                    files = data["files"]
                except KeyError, e:
                    print e
                    print "[!] Exiting: Could not load training set from %s" % dataFile
                    exit(0)
    else:
        print "[!] Exiting: %s is not a valid choice, choose one of" + \
              "\"training\" or \"test\"" % dataSet
        exit(0)

    if poolFile != None:
        try:
            features = sio.loadmat(poolFile)
            pooledFeaturesTrain = features["pooledFeaturesTrain"]
            X = np.transpose(pooledFeaturesTrain, (0,2,3,1))
            numTrainImages = np.shape(X)[3]
            X = np.reshape(X, ((pooledFeaturesTrain.size)/float(numTrainImages), \
                               numTrainImages), order="F")
            scaler = preprocessing.MinMaxScaler()
            # load pooled feature scaler
            #scaler = mlutils.getMinMaxScaler("/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/features/SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_k400_patches_naturalImages_6x6_signPreserveNorm_pooled5.mat")
            scaler.fit(X.T)  # Don't cheat - fit only on training data
            X = scaler.transform(X.T)
            if dataSet == "training":
                pass
            elif dataSet == "test":
                pooledFeaturesTest = features["pooledFeaturesTest"]
                X = np.transpose(pooledFeaturesTest, (0,2,3,1))
                numTestImages = np.shape(X)[3]
                X = np.reshape(X, ((pooledFeaturesTest.size)/float(numTestImages), \
                               numTestImages), order="F")
                X = scaler.transform(X.T)
        except IOError:
            print "[!] Exiting: %s Not Found" % (poolFile)
            exit(0)
        finally:
            features = None
            pooledFeaturesTrain = None
            pooledFeaturesTest = None
                
    #clfFile = "/Users/dew/development/PS1-Real-Bogus/rf/trained/" + \
    #          "RF_n_estimators1000_max_features25_min_samples_leaf1_" + \
    #          "md_20x20_skew4_SignPreserveNorm_with_confirmed1.pkl"
              
    #dataFile = "/Users/dew/development/PS1-Real-Bogus/data/3pi/" + \
    #           "3pi_20x20_realOnly_signPreserveNorm.mat"
    
    #training_inputFile = "/Users/dew/development/PS1-Real-Bogus/data/md/" + \
    #                     "training_realObject_data.csv"
    #testing_inputFile = "/Users/dew/development/PS1-Real-Bogus/data/3pi/" + \
    #                    "3pi_realObject_data.csv"
    
    #plot_dist(training_inputFile, testing_inputFile, alpha2=0.75)
    fileList = []
    for file in files[y==0]:
        fileList.append(str(file).rstrip())
    plot_MDR_vs_mag(classifierFile, X[y==0,:], fileList, infoFile, threshold, color="#04E762")
    
if __name__ == "__main__":
    main()
