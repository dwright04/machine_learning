import sys
sys.path.insert(1, "/Users/dew/development/PS1-Real-Bogus/demos/")
import mlutils
import numpy as np
import scipy.io as sio
sys.path.insert(1, "/Users/dew/development/PS1-Real-Bogus/tools/")
from classify import hypothesisDist
from profiling import visualiseImages

def main():
    ### training data files ###
    path =         "/Users/dew/development/PS1-Real-Bogus/ufldl/sparsefiltering/"
    dataFile =     "3pi_20x20_skew2_signPreserveNorm.mat"
    patchesFile =  "patches_naturalImages_6x6_signPreserveNorm.mat"
    featuresFile = "SF_maxiter100_L1_3pi_20x20_skew2_signPreserveNorm_6x6_"+\
                   "k400_patches_naturalImages_6x6_signPreserveNorm_pooled5.mat"
    
    ### setup constants ###
    imageDim = 20
    patchDim = 6
    poolDim = 5
    numFeatures = stepSize = 400
    
    # load the classifier
    clf = mlutils.getClassifier()
    # load the scaler for raw pixel data
    scaler = mlutils.getScaler("/Users/dew/development/PS1-Real-Bogus/data/"+dataFile)
    # load pooled feature scaler
    minMaxScaler = mlutils.getMinMaxScaler(path+"features/"+featuresFile)
    # get the patches ised to train sparse filter
    patches = mlutils.getPatches("/Users/dew/development/PS1-Real-Bogus/data/"+patchesFile)
    # load the sparse filter
    SF = mlutils.getSparseFilter(numFeatures, patches["patches"], patchesFile)
    # get the trained sparse filter features
    W = np.reshape(SF.trainedW, (SF.k, SF.n), order="F")

    #data = sio.loadmat("/Users/dew/development/PS1-Real-Bogus/data/LSQ/LSQ_fc_20x20_signPreserveNorm.mat")
    data = sio.loadmat("/Users/dew/development/PS1-Real-Bogus/tools/reviewed_data_sets/LSQ_fc_20x20_signPreserveNorm_reviewed_20150306.mat")
    numImages = 1
    y = data["y"]
    data = data["X"]
    m, n = np.shape(data)
    #print data[1,:]
    print m,n
    pred = []
    for i in range(m):
        images = np.zeros((imageDim,imageDim,1,1))
        #print i,data[i,:]
        #vector = scaler.transform(data[i,:].T)
        vector = data[i,:]
        images[:,:,0,0] += np.reshape(vector, (imageDim,imageDim), order="F")

        # convolve these scaled vectors with the learned features
        pooledFeatures = mlutils.convolve_and_pool(images, W, imageDim, patchDim, poolDim, numFeatures, stepSize)
        # reorder and reshape the convolved and pooled images to pass to classifier
        X = np.transpose(pooledFeatures, (0,2,3,1))
        X = np.reshape(X, (int((pooledFeatures.size)/float(numImages)), \
                       numImages), order="F")
        # correctly scale the convolved and pooled features for the specified classifier
        X = minMaxScaler.transform(X.T)
        # calculate the real - bogus value for this object
        realBogus = clf.predict_proba(X)[:,1]
        print realBogus, y[i]
        pred.append(realBogus)

    pred = np.array(pred)
    #y = np.squeeze(np.ones(np.shape(pred)))

    hypothesisDist(y, pred)

    threshold = 0.5
    positives = pred[y==1]
    negatives = pred[y==0]
    print positives
    
    true_pos = np.where(positives >= threshold)[0]
    false_neg = np.where(positives < threshold)[0]
    print false_neg
    
    true_neg = np.where(negatives < threshold)[0]
    false_pos = np.where(negatives >= threshold)[0]

    label = 1
    print y==1
    if label == 1 or 1 in set(y):
        m, n = np.shape(data[y==1])
        print m
        print "[*] %d false negatives." % (len(false_neg))
        print "[*] FNR (MDR) = %.3f" % (len(false_neg)/float(m))
        fnX = data[y==1][false_neg,:]
        fn_pred = pred[y==1][false_neg]
        
        visualiseImages(fnX[:400], fn_pred[:400], False)

if __name__ == "__main__":
    main()