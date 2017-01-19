import pickle, optparse
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from classify import predict
from classify import plot_ROC
from sklearn import preprocessing
from sklearn.metrics import roc_curve

def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(","))

def main():
    
    parser = optparse.OptionParser("[!] usage: python classify.py\n"+\
                                   " -F <data file>\n"+\
                                   " -c <classifier file list [csv]>\n"+\
                                   " -s <data set [optional, default=test]>\n"+\
                                   " -P <list of pooled features files [csv/optional]>")
        
    parser.add_option("-F", dest="dataFile", type="string", \
                      help="specify data file to analyse")
    parser.add_option("-c", dest="classifierFileList", type="string", action="callback", callback=list_callback, \
                      help="specify list of classifiers to compare (comma separated)")
    parser.add_option("-s", dest="dataSet", type="string", \
                      help="specify data set to analyse [optional, default=test]")
    parser.add_option("-P", dest="poolFileList", type="string", action="callback", callback=list_callback, \
                      help="specify list of pooled features files for convolutional classifiers [optional]")


    (options, args) = parser.parse_args()
    dataFile = options.dataFile
    classifierFileList = options.classifierFileList
    dataSet = options.dataSet
    poolFileList = options.poolFileList
 
    FPRs = [0.01, 0.05, 0.1]
    default_ticks = [0, 0.05, 0.10, 0.25]
    colors = ["#FF0066", "#66FF33", "#3366FF"]
    
    fig = plt.figure()
    font = {"size": 26}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=22)

    plt.xlabel("Missed Detection Rate (MDR)")
    plt.ylabel("False Positive Rate (FPR)")
    plt.yticks([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    plt.ylim((0,1.05))
    
    
    poolFileCounter = 0

    if dataFile == None or classifierFileList == None:
        print parser.usage
        exit(0)
    if dataSet == None:
        dataSet = "test"

    for i,classifierFile in enumerate(classifierFileList):
        data = sio.loadmat(dataFile)
        if dataSet == "training":
            y = data["y"]
            X = data["X"]
        else:
            y = data["testy"]
            X = data["testX"]
        try:
            pred = predict(classifierFile, X)
        except ValueError:
            poolFile = poolFileList[poolFileCounter]
            poolFileCounter += 1
            try:
                features = sio.loadmat(poolFile)
                pooledFeaturesTrain = features["pooledFeaturesTrain"]
                X = np.transpose(pooledFeaturesTrain, (0,2,3,1))
                numTrainImages = np.shape(X)[3]
                X = np.reshape(X, ((pooledFeaturesTrain.size)/float(numTrainImages), \
                               numTrainImages), order="F")
                scaler = preprocessing.MinMaxScaler()
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
                pred = predict(classifierFile, X)
            except IOError:
                print "[!] Exiting: %s Not Found" % (poolFile)
                exit(0)
            finally:
                features = None
                pooledFeaturesTrain = None
                pooledFeaturesTest = None
        color = colors[i]
        fpr, tpr, thresholds = roc_curve(y, pred)

        plt.plot(1-tpr, fpr, "k-", lw=5)
        plt.plot(1-tpr, fpr, color=color, lw=4)

        ticks = []
        FoMs = []
        decisionBoundaries = []

        for FPR in FPRs:
            FoMs.append(1-tpr[np.where(fpr<=FPR)[0][-1]])
            decisionBoundaries.append(thresholds[np.where(fpr<=FPR)[0][-1]])
        for i,FoM in enumerate(FoMs):
            print "[+] FoM : %.3f | decision boundary : %.3f " % (FoM, decisionBoundaries[i])
            plt.plot([x for x in np.arange(0,FoM+1e-3,1e-3)], \
                     FPRs[i]*np.ones(np.shape(np.array([x for x in np.arange(0,FoM+1e-3,1e-3)]))), \
                     "k--", lw=3)
            
            plt.plot(FoM*np.ones(np.shape([x for x in np.arange(0,FPRs[i]+1e-3, 1e-3)])), \
                     [x for x in np.arange(0,FPRs[i]+1e-3, 1e-3)], "k--", lw=3)
            if round(FoM,2) in default_ticks:
                default_ticks.remove(round(FoM,2))
                ticks.append(FoM)
            else:
                ticks.append(FoM)
                plt.xticks(default_ticks+ticks, rotation=70)

        locs, labels = plt.xticks()
        plt.xticks(locs, map(lambda x: "%.3f" % x, locs))
    plt.show()


if __name__ == "__main__":
    main()
