import os
import sys
import cv2
import numpy as np

def get_hog() :
    winSize = (100,100)
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (10,10)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)

    return hog

def load_trainData(image_path):
    trainData = []
    trainLabel = []
    i = 0
    for path in sorted(os.listdir(image_path)):

        files = os.listdir(image_path +'/'+ path)
        for f in files:
             if (f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg')):
                 trainData.append(image_path +'/'+ path +'/' + f)
                 trainLabel.extend([x for x in np.repeat(i,4)])

        print ("{} -> {} ".format(path,i))
        i = i+1

    return trainData,trainLabel


if __name__ == '__main__':

    image_path = "data/train"
    trainData,trainLabel  = load_trainData(image_path)

    # HoG feature descriptor
    hog = get_hog();
    hog_descriptors = []
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setType(cv2.ml.SVM_C_SVC)

    k=0
    for data in trainData:
        img = cv2.imread(data,0)
        resized_img = cv2.resize(img,(100,100),interpolation = cv2.INTER_CUBIC)
        gauss_img = cv2.GaussianBlur(resized_img,(9,9),0)
        th = cv2.adaptiveThreshold(gauss_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        		cv2.THRESH_BINARY_INV,11,2)
        cv2.imwrite('data/th/'+str(k)+'.jpg',th)
        hog_descriptors.append(hog.compute(th))
        rows,cols = resized_img.shape
        for i in [1,2,3]:
            M   = cv2.getRotationMatrix2D((cols/2,rows/2),i*90,1)
            dst = cv2.warpAffine(th,M,(cols,rows))
            hog_descriptors.append(hog.compute(dst))
            cv2.imwrite('data/th/'+str(k)+'_'+str(i)+'.jpg',dst)

        k = k+1

    svm.trainAuto(np.array(hog_descriptors,np.float32), cv2.ml.ROW_SAMPLE, np.array(trainLabel))
    svm.save('svm_data.dat')

    test = cv2.imread("data/img.jpg",0)
    test_hog = hog.compute(test)
    re = svm.predict(np.array(test_hog,np.float32).reshape(-1,3249))
    print re
