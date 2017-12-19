import numpy as np
import cv2
import sys
import imutils
from pylsd import lsd
from random import randint
from segment import *

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

def draw_result_boxes(img,boxes):
    text = ['v_source','capacitor','ground','diode','resistor','inductor']
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ((x,y,w,h),idx) in boxes:
    		cv2.putText(img, text[idx] ,(x-5,y-5),font,0.7,(0,255,0),1,cv2.LINE_AA)
    		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

if __name__ == "__main__":

    src = cv2.imread("data/x11.jpg")
    src = imutils.resize(src,width=640)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    ## branch, endpoint operations
    img = cv2.GaussianBlur(gray,(9,9),0)
    th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    		cv2.THRESH_BINARY_INV,11,2)
    th2 = th.copy()
    bw = thinning(th)
    cv2.imwrite("data/skel.pgm",bw)
    ends,branch,branch_ok = skeleton_points(bw)

    ## detection of ground, capacitor, v_source
    v_pairs,h_pairs = lines_between_ends(ends)
    v_boxes = box_between_ends(v_pairs)
    h_boxes = box_between_ends(h_pairs)
    boxes = v_boxes + h_boxes

    ## segmentation operations
    ## remove founded symbols and connection lines
    for ((x,y,w,h),idx) in boxes:
    	th[y:y+h,x:x+w] = 0
    cv2.imwrite("removed.pgm",th)
    ## detect vert and hori lines then remove them from binary image
    lsd_lines = lsd(th)
    for line in lsd_lines:
    	x1,y1,x2,y2,w = line
    	angle = np.abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
    	if (angle<105 and angle>75) or angle>160 or angle<20:
            cv2.line(th,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,0),6)

    kernel = np.ones((11,11),np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    rects = []
    # Find Blobs on image
    cnts = cv2.findContours(closing.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    for c in cnts:
        if cv2.contourArea(c)<80:
            continue
        else:
            x,y,w,h = cv2.boundingRect(c)
            maxedge = max(w,h)
            x,y = int(((2*x+w)-maxedge)/2),int(((2*y+h)-maxedge)/2)
            rects.append([x-10,y-10,x+maxedge+10,y+maxedge+10])

    rects = non_max_suppression_fast(np.array(rects,dtype=float),0.1)


    ## HOG+SVM prediction
    svm = cv2.ml.SVM_load("svm_data.dat")
    hog = get_hog();
    for x,y,x2,y2 in rects:
    	region = cv2.resize(th2[y:y2,x:x2],(100,100),interpolation = cv2.INTER_CUBIC)
    	hog_des = hog.compute(region)
    	_,result = svm.predict(np.array(hog_des,np.float32).reshape(-1,3249))
        idx = int(result[0][0])+3
        boxes.append([[int(x),int(y),int(x2-x),int(y2-y)],idx])

    draw_result_boxes(src,boxes)
    cv2.imwrite("result.jpg",src)
    cv2.imshow("org", src)
    cv2.waitKey(0)
