#!/usr/bin/python

import pickle, sys, pprint
import scipy.io as sio
import numpy as np
from NeuralNet import NeuralNetPerform
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.metrics import roc_curve, f1_score
from sklearn.cross_validation import KFold

def crossValidate(arch, lambdaGrid, train_x, train_y, dataFile):
    font = {"size": 28}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=24)
    acceptableFPR = 0.01
    kf = KFold(len(train_y), n_folds=5, indices=False)
    # get the number of curves, for plotting line colours and
    # do some funky stuff to get the color map for plotting later
    """
    values = range(len(lambdaGrid))
    cmap = plt.get_cmap("gist_rainbow")
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    """
    path = "cv/"
    """
    plt.title(dataFile.split("/")[-1].split(".")[0]+": gamma=%.3f" % (gamma))
    plt.xlabel("Missed Detection Rate (MDR)")
    plt.ylabel("False Positive Rate (FPR)")
    """

    for i,LAMBDA in enumerate(lambdaGrid):
        FoMs = []
        fold = 1
        FNR = np.ones((1000, 5))
        FPR = np.ones((1000, 5))
        for train, test in kf:
            savedNNfile = path+"trainedNet_NerualNet_%s_arch%d_lambda%f_fold%d.mat" % \
                          (dataFile.split("/")[-1].split(".")[0], arch, LAMBDA, fold)
            test_x, test_y = train_x[test], train_y[test]
            test_x = test_x.T
            test_y = test_y[np.newaxis]
            try:
                nn = NeuralNetPerform(test_x, test_y, saveFile=savedNNfile)
                print nn._architecture
            except:
                print "    [-] Could not find saved classifier."
                continue
            hypothesis = nn.predict_proba(test_x)[:,1]
            fpr, tpr, precision = nn.calculatePerformanceIndicators(test_x, test_y, hypothesis, 0.5)
            print "    [+] F1 Score: %.3f" % nn.calculateF1Score(precision, tpr)
            fpr, fnr, opt_threshold,  fom = nn.plotROCCurve(test_x, test_y, acceptableFPR, \
                                                            tolerance=1e-3, plot=False)
            print "    [+] FoM: %.3f" % fom
            FoMs.append(fom)
            print "    [+] Threshold: %.3f" % opt_threshold
            # get the color for this line
            #colorVal = scalarMap.to_rgba(values[i])
            #plt.plot(1-tprSVM, fprSVM, label=str(C), color=colorVal, linewidth=2.0)
        
            FNR[:,fold-1] = FNR[:,fold-1] * fnr
            FPR[:,fold-1] = FPR[:,fold-1] * fpr
            test_x = None
            test_y = None

            if LAMBDA == 0.3:
                color = "#3366FF"
                if fold == 1:
                    print test_x
                    test_x, test_y = train_x[train][:0.5*len(train)], train_y[train][:0.5*len(train)]
                    test_x = test_x.T
                    print np.shape(test_x)
                    print np.shape(train_x[test].T)
                    test_x = np.concatenate((test_x, train_x[test].T), axis=1)
                    print np.shape(test_x)
                    test_y = test_y[np.newaxis]
                    test_y = np.concatenate((test_y, train_y[test][np.newaxis]), axis=1)
                    hypothesis = nn.predict(test_x)
                    fpr, tpr, precision = nn.calculatePerformanceIndicators(test_x, test_y, hypothesis, 0.5)
                    print "    [+] F1 Score: %.3f" % nn.calculateF1Score(precision, tpr)
                    fpr, fnr, opt_threshold,  fom = nn.plotROCCurve(test_x, test_y, acceptableFPR, \
                                                                   tolerance=5e-4, plot=False)
                    label="overfit"
                    color="#9933FF"
                    plt.plot(fnr, fpr, color="k", lw=5, zorder=50)
                    plt.plot(fnr, fpr, color=color, lw=4, zorder=51, label=label)
            elif LAMBDA == 100:
                color="#66FF33"
            elif LAMBDA == 5:
                color="#FF0066"
            plt.plot(fnr, fpr, color=color, lw=3, alpha=0.5)
            if fold == 5:
                label = str(LAMBDA)
                plt.plot(np.mean(FNR, axis=1), np.mean(FPR, axis=1), color="k", lw=5, zorder=100)
                plt.plot(np.mean(FNR, axis=1), np.mean(FPR, axis=1), color=color, lw=4, zorder=200, label=label)
            fold +=1
            #if fold ==5 :
            #    break

    
        print "[*] arch: %d, lambda: %f mean FoM: %.3f" % \
               (arch, LAMBDA, np.mean(FoMs))
                   

    plt.yticks([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.ylim((-0.001,0.51))
    plt.xlim((-0.001,0.51))
    plt.plot(fnr, 0.01*np.ones(np.shape(fnr)), "k--", lw=2, zorder=300)
    plt.legend(title="lambda")
    plt.xlabel("Missed Detection Rate (MDR)")
    plt.ylabel("False Positive Rate (FPR)")
    plt.show()
    """
    plt.plot(1-tprSVM, 0.01*np.ones(np.shape(fprSVM)), "k--")
    plt.plot(1-tprSVM, 0.05*np.ones(np.shape(fprSVM)), "k--")
    plt.plot(1-tprSVM, 0.1*np.ones(np.shape(fprSVM)), "k--")
    plt.legend(title="C")
    plt.show()
    """

def plotFinalResult(arch, lambdaGrid, test_x, test_y, dataFile):

    values = range(len(lambdaGrid))
    cmap = plt.get_cmap("gist_rainbow")
    cNorm = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    
    font = {"size": 18}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=14)
    
    plt.xlabel("Missed Detection Rate (MDR)")
    plt.ylabel("False Positive Rate (FPR)")
    plt.yticks([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    plt.ylim((0,1.05))
    plt.xticks([0, 0.01, 0.033, 0.05, 0.055, 0.203, 0.25], rotation=70)
    
    FoMs = []
    for i,LAMBDA in enumerate(lambdaGrid):
        savedNNfile = "NerualNet_%s_arch%d_lambda%.6f.mat" % \
            (dataFile.split("/")[-1].split(".")[0], arch, LAMBDA)

        nn = pickle.load(open(savedNNfile.split(".")[0]+"."+savedNNfile.split(".")[1]+".pkl", "rb"))
        nn.saveNetwork(savedNNfile)
        nn = NeuralNetPerform(test_x, test_y, saveFile=savedNNfile)

        hypothesis = nn.predict(test_x)

        fpr, tpr, precision = nn.calculatePerformanceIndicators(test_x, test_y, hypothesis, 0.5)
        print nn.calculateF1Score(precision, tpr)
        acceptableFPR = 0.01
        fpr, fnr, opt_threshold, fom = nn.plotROCCurve(test_x, test_y, acceptableFPR, tolerance=1e-4, plot=False)
        print fom
        #print fpr[np.where((1-tpr) <= 0.11)[0]]
        #print (1-tpr)[np.where((1-tpr) <= 0.11)[0]]
        
        print opt_threshold
        
        # get the color for this line
        colorVal = scalarMap.to_rgba(values[i])

        print np.shape(fnr)
        print np.shape(fpr)

        fnr = np.array(fnr)
        fpr = np.array(fpr)
        
        print fnr[np.where(fnr<=0.01)[0]]
       
        plt.plot(fnr, fpr, color="k", linewidth=5.0)
        plt.plot(fnr, fpr, color="#9933FF", linewidth=4.0)
            

    intersect10 = np.shape(fnr[np.where(fnr <= 0.0334801762115)])[0]
    
    plt.plot(fnr[np.where(fnr <= 0.0334801762115)], \
             0.1*np.ones(np.shape(fpr[:intersect10])), "k--", linewidth=2.0)
             
    plt.plot(0.0334801762115*np.ones(len(fpr[np.squeeze(np.where(fpr <= 0.1))])), \
            fpr[np.squeeze(np.where(fpr <= 0.1))], "k--", linewidth=2.0)
    
    intersect5 = np.shape(fnr[np.where(fnr <= 0.0546255506608)])[0]
                      
    plt.plot(fnr[np.where(fnr <= 0.0546255506608)], \
            0.05*np.ones(np.shape(fpr[:intersect5])), "k--", linewidth=2.0)
                               
    plt.plot(0.0546255506608*np.ones(len(fpr[np.squeeze(np.where(fpr <= 0.05))])), \
            fpr[np.squeeze(np.where(fpr <= 0.05))], "k--", linewidth=2.0)

    intersect1 = np.shape(fnr[np.where(fnr <= 0.20264317)])[0]
                                        
    plt.plot(fnr[np.where(fnr <= 0.20264317)], \
            0.01*np.ones(np.shape(fpr[:intersect1])), "k--", linewidth=2.0)
                                                 
    plt.plot(0.20264317*np.ones(len(fpr[np.squeeze(np.where(fpr <= 0.01))])), \
            fpr[np.squeeze(np.where(fpr <= 0.01))], "k--", linewidth=2.0)

    plt.show()

def main():

    """
        In this case the models have already been 5-fold cross validated using sklean
        GridSearchCV on the combined trainign and CV sets in my dataset files.
        
        Here we just pass in the X and y we wish to test on.
    """
    #lambdaGrid = [1,10, 100]
    #lambdaGrid = [0.3, 5, 100]
    lambdaGrid = [1.0,5.0]
    arch = 200
    #dataFile = "/Users/dew/myscripts/machine_learning/analysis/dataSets/trainingSets/md/" + \
    #           "md_20x20_skew4_SignPreserveNorm_final_tti.mat"
    #dataFile = "/Users/dew/myscripts/machine_learning/analysis/dataSets/trainingSets/md/" + \
    #           "md_20x20_skew4_SignPreserveNorm_scannedGarbageNoUnclean_final_tti.mat"
    #dataFile = "/Users/dew/myscripts/machine_learning/analysis/dataSets/trainingSets/md/" + \
    #           "md_20x20_skew4_SignPreserveNorm_final_tti2.mat"
    #dataFile = "/Users/dew/myscripts/machine_learning/analysis/dataSets/trainingSets/md/" + \
    #           "md_20x20_skew4_SignPreserveNorm_scannedGarbageNoUnclean_final_tti2.mat"
    #dataFile = "../md_20x20_skew4_SignPreserveNorm_with_confirmed1.mat"

    dataFile = "/Users/dew/development/PS1-Real-Bogus/data/3pi/3pi_20x20_signPreserveNorm.mat"
    ####### Load the data  #######
    data = sio.loadmat(dataFile)
    train_x = data["X"]
    train_y = np.squeeze(data["y"])
    #train_x = np.concatenate((data["X"], data["validX"]))
    #train_y = np.squeeze(np.concatenate((data["y"], data["validy"])))
    #mu = np.mean(train_x, axis=0)
    #sigma = np.std(train_x, axis=0)
    #train_x = train_x - mu
    #train_x = train_x / sigma
    test_x = data["testX"]
    test_y = data["testy"]
    #test_x = test_x - mu
    #test_x = test_x / sigma

    crossValidate(arch, lambdaGrid, train_x, train_y, dataFile)
    #plotFinalResult(arch, lambdaGrid, test_x.T, test_y[np.newaxis], dataFile)

if __name__ == "__main__":
    main()
