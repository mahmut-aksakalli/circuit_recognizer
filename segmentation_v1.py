from scipy import weave
import numpy as np
import cv2
import sys
import imutils
from random import randint

## capacitor,v_source

def _thinningIteration(im, iter):
	I, M = im, np.zeros(im.shape, np.uint8)
	expr = """
	for (int i = 1; i < NI[0]-1; i++) {
		for (int j = 1; j < NI[1]-1; j++) {
			int p2 = I2(i-1, j);
			int p3 = I2(i-1, j+1);
			int p4 = I2(i, j+1);
			int p5 = I2(i+1, j+1);
			int p6 = I2(i+1, j);
			int p7 = I2(i+1, j-1);
			int p8 = I2(i, j-1);
			int p9 = I2(i-1, j-1);
			int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
			         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
			         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
			         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
			if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
				M2(i,j) = 1;
			}
		}
	}
	"""

	weave.inline(expr, ["I", "iter", "M"])
	return (I & ~M)

def thinning(src):
	dst = src.copy() / 255
	prev = np.zeros(src.shape[:2], np.uint8)
	diff = None

	while True:
		dst = _thinningIteration(dst, 0)
		dst = _thinningIteration(dst, 1)
		diff = np.absolute(dst - prev)
		prev = dst.copy()
		if np.sum(diff) == 0:
			break

	return dst * 255

def skeleton_points(skel):
	# make out input nice, possibly necessary
	skel = skel.copy()
	skel[skel!=0] = 1
	skel = np.uint8(skel)

	# kernel for end and branch points
	kernel = np.uint8([[1,  1, 1],
	                   [1, 10, 1],
	                   [1,  1, 1]])

	# kernel for branch points
	kernel2 = np.uint8([[0,  1, 0],
	                    [1, 10, 1],
	                    [0,  1, 0]])

	filtered = cv2.filter2D(skel,-1,kernel)
	filtered2 =cv2.filter2D(skel,-1,kernel2)

	out = np.zeros_like(skel)
	out[np.where(filtered > 12)] =255
	cv2.imwrite("branch.pgm",out)

	ends   = np.where(filtered == 11)
	branch = np.where(filtered > 12)
	branch_ok = np.where(filtered2 > 12)

	return ends,branch,branch_ok

def point_to_line_dist(point, line):
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)
   # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
        np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
        np.linalg.norm(unit_line)
    )

    diff = (
        (norm_unit_line[0] * (point[0] - line[0][0])) +
        (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist

def lines_between_ends(ends):
	v_pairs = []
	h_pairs = []
	img  = cv2.imread("branch.pgm",0)
	skel = cv2.imread("skel.pgm",0)
	## limit angles of pair lines
	for i in xrange(ends[0].size):
		x1,y1 = (ends[1][i],ends[0][i])
		for j in xrange(ends[0].size):
			if( i!=j and j>i):
				x2,y2 = (ends[1][j],ends[0][j])
				dist  = np.sqrt((x1-x2)**2+(y1-y2)**2)
				miny,maxy = sorted([y1,y2])
				minx,maxx = sorted([x1,x2])
				angle = np.abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
				if (angle<105 and angle>75 and dist<100 and dist>5):
					q = np.where(img[miny:maxy,minx-5:maxx+5] == 255)
					if len(q[0])>0:
						v_pairs.append([x1,y1,x2,y2])
				elif ((angle>170 or angle<10) and dist<100 and dist>5):
					q = np.where(img[miny-5:maxy+5,minx:maxx] == 255)
					if len(q[0])>0:
						h_pairs.append([x1,y1,x2,y2])
	return np.array(v_pairs),np.array(h_pairs)

def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]

	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]

	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
	(tr, br) = rightMost

	return np.array([tl,tr,br,bl])

def box_between_ends(lines):
	boxes = []

	for i in xrange(len(lines)):
		midx1,midy1 = ((lines[i,0]+lines[i,2])/2,(lines[i,1]+lines[i,3])/2)
		for j in xrange(len(lines)):
			if( i!=j and j>i):
				midx2,midy2 = ((lines[j,0]+lines[j,2])/2,(lines[j,1]+lines[j,3])/2)
				dist  = np.sqrt((midx1-midx2)**2+(midy1-midy2)**2)
				if dist<40:
					tl,tr,br,bl = order_points(np.array([(lines[i,0],lines[i,1]),\
								(lines[i,2],lines[i,3]),(lines[j,0],lines[j,1]),\
								(lines[j,2],lines[j,3])]))
					n =  np.unique([tl,tr,br,bl],axis=0)
					if n.shape[0] ==4:
						h1=np.sqrt((lines[i,0]-lines[i,2])**2+(lines[i,1]-lines[i,3])**2)
						h2=np.sqrt((lines[j,0]-lines[j,2])**2+(lines[j,1]-lines[j,3])**2)
						small,big = sorted([h1,h2])
						if (big/small)>=1 and ((big/small)<=1.2):
							idx = 1
						elif (big/small)>1.2:
							idx = 0
						tl = [tl[0]-5,tl[1]-5]
						tr = [tr[0]+5,tr[1]-5]
						br = [br[0]+5,br[1]+5]
						bl = [bl[0]-5,bl[1]+5]
						box = cv2.boundingRect(np.array([tl,tr,br,bl]))
						boxes.append([box,idx])
	return boxes

if __name__ == "__main__":

	src = cv2.imread("x2.jpg")
	src = imutils.resize(src,width=640)
	gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

	img = cv2.GaussianBlur(gray,(9,9),0)
	th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
			cv2.THRESH_BINARY_INV,11,2)
	bw = thinning(th)
	cv2.imwrite("skel.pgm",bw)
	ends,branch,branch_ok = skeleton_points(bw)

	v_pairs,h_pairs = lines_between_ends(ends)
	v_boxes = box_between_ends(v_pairs)
	h_boxes = box_between_ends(h_pairs)
	if len(v_boxes)>0 and len(h_boxes)>0:
		boxes = np.concatenate((v_boxes,h_boxes),axis=0)
	elif len(v_boxes)>0:
		boxes = v_boxes
	elif len(h_boxes)>0:
		boxes = h_boxes
	else:
		boxes = []

	for i in xrange(branch[0].size):
		cv2.circle(src,(branch[1][i],branch[0][i]),3,(255,0,0),-1)
	for i in xrange(branch_ok[0].size):
		cv2.circle(src,(branch_ok[1][i],branch_ok[0][i]),3,(0,0,255),-1)
	for i in xrange(ends[0].size):
		cv2.circle(src,(ends[1][i],ends[0][i]),2,(0,255,0),-1)


	for i in xrange(len(v_pairs)):
	    pt1 = (int(v_pairs[i, 0]), int(v_pairs[i, 1]))
	    pt2 = (int(v_pairs[i, 2]), int(v_pairs[i, 3]))
	    cv2.line(src, pt1, pt2, (randint(0,255), randint(0,255),randint(0,255)), 2)

	for i in xrange(len(h_pairs)):
	    pt1 = (int(h_pairs[i, 0]), int(h_pairs[i, 1]))
	    pt2 = (int(h_pairs[i, 2]), int(h_pairs[i, 3]))
	    cv2.line(src, pt1, pt2, (randint(0,255), randint(0,255),randint(0,255)), 2)

	for ((x,y,w,h),idx) in boxes:
		if idx == 0:
			cv2.rectangle(src,(x,y),(x+w,y+h),(0,255,0),1)
		elif idx == 1:
			cv2.rectangle(src,(x,y),(x+w,y+h),(0,255,255),1)

	cv2.imshow("thinning", src)
	cv2.waitKey()
