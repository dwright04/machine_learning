import sys, multiprocessing, pickle
import scipy.io as sio
sys.path.append("../tools/")
import multiprocessingUtils
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

class trainRF(multiprocessingUtils.Task):
    def __init__(self, X, y, dataFile, fold, rf):
        self.X = X
        self.y = np.squeeze(y)
        self.dataFile = dataFile
        self.fold = fold
        self.rf = rf
        self.n_estimators= rf.n_estimators
        self.max_features = rf.max_features
        self.min_samples_leaf = rf.min_samples_leaf

    def __call__(self):
        self.rf.fit(self.X, self.y)
        outputFile = open("cv/RF_n_estimators"+str(self.n_estimators)+\
                          "_max_features"+str(self.max_features)+"_min_samples_leaf"+str(self.min_samples_leaf)+\
                          "_"+self.dataFile.split("/")[-1].split(".")[0]+"_fold_"+str(self.fold)+".pkl", "wb")
        pickle.dump(self.rf, outputFile)
        outputFile.close()
        return 0
    
    def __str__(self):
        return "### Training Random Forest with n_estimator = %f, max_features = %f, min_samples_leaf = %d ###" \
               % (self.n_estimators, self.max_features, self.min_samples_leaf)

def main(argv = None):
    
    if argv is None:
        argv = sys.argv
    
    if len(argv) != 3:
        sys.exit("Usage: multicore_rf_gridSearch.py <.mat file> <number of folds>")

    dataFile = argv[1]
    nFolds   = int(argv[2])

    data = sio.loadmat(dataFile)

    X = data["X"]
    y = data["y"]

    n_estimatorsGrid = [1000] # ntree in Brink et al. 2013
    #max_featuresGrid = [2,4,10,25,50,100] # mtry in Brink et al. 2013
    #min_samples_leafGrid = [1,2,4]
    #max_featuresGrid = [2,4,25,50] # mtry in Brink et al. 2013
    max_featuresGrid = [20,25,50]
    min_samples_leafGrid = [1,2,4]

    kf = KFold(m, n_folds=nFolds)
    for n_estimators in n_estimatorsGrid:
        for max_features in max_featuresGrid:
            for min_samples_leaf in min_samples_leafGrid:
                taskList = []
                fold = 1
                for train, test in kf:
                    taskList.append(trainRF(X, y, dataFile, fold, \
                                            RandomForestClassifier(n_estimators=n_estimators, \
                                                                   max_features=max_features, \
                                                                   min_samples_leaf=min_samples_leaf)))
         fold += 1

    cpu_count = multiprocessing.cpu_count()
    
    print "%d available CPUs.\n" % (cpu_count)

    multiprocessingUtils.multiprocessTaskList(taskList, cpu_count)

if __name__ == "__main__":
    main()
