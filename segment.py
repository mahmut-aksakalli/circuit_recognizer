from scipy import weave
import numpy as np
import cv2

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

	filtered = cv2.filter2D(skel,-1,kernel)

	out = np.zeros_like(skel)
	out[np.where(filtered > 12)] =255

	ends   = np.where(filtered == 11)
	return ends

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
	skel = cv2.imread("data/skel.pgm",0)

	for i in xrange(ends[0].size):
		x1,y1 = (ends[1][i],ends[0][i])
		for j in xrange(ends[0].size):
			if( i!=j and j>i):
				x2,y2 = (ends[1][j],ends[0][j])
				dist  = np.sqrt((x1-x2)**2+(y1-y2)**2)
				miny,maxy = sorted([y1,y2])
				minx,maxx = sorted([x1,x2])
				angle = np.abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
				## for vertical line pairs
				if (angle<105 and angle>75 and dist<100 and dist>5):
					q1 = np.where(skel[miny-5:maxy+5,minx-5:maxx+5] == 255)
					point_count = 0
					for k in xrange(len(q1[0])):
						d1 = point_to_line_dist(np.array([minx-5+q1[1][k],miny-5+q1[0][k]]),\
								np.array([[x1,y1],[x2,y2]]))
						point_count = point_count+1 if d1<2 else point_count

					if point_count>dist*0.7:
						v_pairs.append([x1,y1,x2,y2])
				## for horizontal line pairs
				elif ((angle>170 or angle<15) and dist<100 and dist>5):
					q1 = np.where(skel[miny-5:maxy+5,minx-5:maxx+5] == 255)
					point_count = 0
					for k in xrange(len(q1[0])):
						d1 = point_to_line_dist(np.array([minx-5+q1[1][k],miny-5+q1[0][k]]),\
								np.array([[x1,y1],[x2,y2]]))
						point_count = point_count+1 if d1<2 else point_count

					if point_count>dist*0.7:
						h_pairs.append([x1,y1,x2,y2])

	return np.array(v_pairs),np.array(h_pairs)

def order_points(pts):
	c = len(pts)
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]

	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[c-2:, :]

	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
	(tr, br) = rightMost

	return np.array([tl,tr,br,bl])

def box_between_ends(lines):
	boxes = []
	linepairs = []

	## find all nearest pair of lines
	lookup = np.zeros((len(lines),len(lines)),dtype=np.int)
	lookup = lookup-1
	for i in xrange(len(lines)):
		midx1,midy1 = ((lines[i,0]+lines[i,2])/2,(lines[i,1]+lines[i,3])/2)
		for j in xrange(len(lines)):
			if( i!=j and j>i):
				midx2,midy2 = ((lines[j,0]+lines[j,2])/2,(lines[j,1]+lines[j,3])/2)
				dist  = np.sqrt((midx1-midx2)**2+(midy1-midy2)**2)
				if dist<60:
					c = np.where(lookup == j)
					if len(c[0])==0:
						cx = np.where(lookup == i)
						if len(cx[0])==0:
							lookup[i,j] = j
						elif len(cx[0])>0:
							lookup[cx[0][0],j] = j
					elif len(c[0])>0:
						lookup[c[0][0],i] = i

	for k in xrange(lookup.shape[0]):
		c = np.where(lookup[k,:] != -1)
		## if only two line exist which means capacitor or source
		if len(c[0]) == 1:
			i,j = k, c[0][0]
			tl,tr,br,bl = order_points(np.array([(lines[i,0],lines[i,1]),\
						(lines[i,2],lines[i,3]),(lines[j,0],lines[j,1]),\
						(lines[j,2],lines[j,3])]))
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
		## if there is 3 lines which means certainly ground
		elif len(c[0]) == 2:
			x,y,z = k,c[0][0],c[0][1]
			tl,tr,br,bl = order_points(np.array([(lines[x,0],lines[x,1]),\
						(lines[x,2],lines[x,3]),(lines[y,0],lines[y,1]),\
						(lines[y,2],lines[y,3]),(lines[z,0],lines[z,1]),\
						(lines[z,2],lines[z,3])]))
			tl = [tl[0]-10,tl[1]-10]
			tr = [tr[0]+10,tr[1]-10]
			br = [br[0]+15,br[1]+15]
			bl = [bl[0]-15,bl[1]+15]
			box = cv2.boundingRect(np.array([tl,tr,br,bl]))
			boxes.append([box,2])

	return boxes

def non_max_suppression_fast(boxes,
                             overlapThresh = 0.1):

	if len(boxes) == 0:
	    return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute area of the bounding boxes and
	# sort by bottom-right y-coord. of the bounding boxes
	area = ( x2-x1+1 ) * ( y2-y1+1 )
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs)-1
		i = idxs[last]
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		c = np.where(overlap > overlapThresh)
		## if no overlapping box, return itself
		if len(c[0])==0:
			pick.append(boxes[i])
		## if there is overlapping box, return combination of boxes
		elif len(c[0])>0:
			group = np.concatenate(([last],c[0]))
			minx = np.amin(x1[idxs[group]])
			miny = np.amin(y1[idxs[group]])
			maxx = np.amax(x2[idxs[group]])
			maxy = np.amax(y2[idxs[group]])
			pick.append([minx,miny,maxx,maxy])

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# integer data type
	return np.array(pick).astype("int")

def remove_parallel_lines(lsd_lines,axis):
	minThresh = 30
	lines = []
	result = []

	## axis = 2 , means vertical lines
	if axis == 2:
		for line in lsd_lines:
			x1,y1,x2,y2,w = line
			angle = np.abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
			if angle<100 and angle>80:
				lines.append(line)
	## axis = 1 , means horizontal lines
	elif axis == 1:
		for line in lsd_lines:
			x1,y1,x2,y2,w = line
			angle = np.abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
			if angle>170 or angle<10:
				lines.append(line)
	## find all nearest pair of lines
	lookup = np.zeros((len(lines),len(lines)),dtype=np.int)
	lookup = lookup-1
	i = 0
	for line in lines:
		j = 0
		for test in lines:
			if i != j and j > i:
				 d1 = np.sqrt((test[0]-line[0])**2+(test[1]-line[1])**2)
				 d2 = np.sqrt((test[0]-line[2])**2+(test[1]-line[3])**2)
				 d3 = np.sqrt((test[2]-line[0])**2+(test[3]-line[1])**2)
				 d4 = np.sqrt((test[2]-line[2])**2+(test[3]-line[3])**2)
				 minDist = sorted([d1,d2,d3,d4])[0]
				 if minDist < minThresh:
					c = np.where(lookup == j)
					if len(c[0])==0:
						cx = np.where(lookup == i)
						if len(cx[0])==0:
							lookup[i,j] = j
						elif len(cx[0])>0:
							lookup[cx[0][0],j] = j
					elif len(c[0])>0:
						lookup[c[0][0],i] = i
			j = j+1
		i = i+1

	## combine all nearest lines
	ax1 = axis-1
	ax2 = axis+1
	for i in xrange(lookup.shape[0]):
		c = np.where(lookup[i,:] != -1)
		y = []
		points = []
		a = 0
		if len(c[0])>0:
			for j in xrange(len(c[0])):
				index = c[0][j]
				points.append(lines[index])
				y.append(lines[index][ax1])
				y.append(lines[index][ax2])

			points.append(lines[i])
			y.append(lines[i][ax1])
			y.append(lines[i][ax2])
			miny = sorted(y)[0]
			maxy = sorted(y,reverse=True)[0]

			for p in points:
				if p[ax1] == miny or p[ax1] == maxy:
					if (a == 0):
						t1 = [int(p[0]),int(p[1])]
						a = a + 1
					elif (a == 1):
						t2 = [int(p[0]),int(p[1])]
						break
				if p[ax2] == miny or p[ax2] == maxy:
					if a == 0:
						t1 = [int(p[2]),int(p[3])]
						a = a + 1
					elif a == 1:
						t2 = [int(p[2]),int(p[3])]
						break

			result.append([t1[0],t1[1],t2[0],t2[1],1])
		elif len(c[0]) ==0:
			n = np.count_nonzero(lookup[:,i] != -1)
			if n == 0:
				result.append([lines[i][0],lines[i][1],lines[i][2],lines[i][3],1])

	return result
