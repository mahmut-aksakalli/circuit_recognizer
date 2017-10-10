import cv2
import math
import imutils
import numpy as np
from pylsd import lsd
from random import randint

def gradient_x(I):
	filtered_img = np.zeros(I.shape,dtype=np.float)
	i=1
	while(i<len(I)-1):
		j=1
		while(j<len(I[0])-1):
			diff = I[i][j+1]+(-1*I[i][j-1])
			filtered_img[i][j] = int(round(diff))
			j+=1
		i+=1
	return filtered_img

def gradient_y(I):
	filtered_img = np.zeros(I.shape)
	i=1
	while(i<len(I)-1):
		j=1
		while(j<len(I[0])-1):
			diff = I[i+1][j]+(-1*I[i-1][j])
			filtered_img[i][j] = int(round(diff))
			j+=1
		i+=1
	return filtered_img

def gauss(x,y,sigma):
	return math.exp((-((x**2)+(y**2)))/(2*(sigma**2)))

def smooth(img,s,x,y,kernel_size):
	khs = kernel_size[0]/2
	temp = 0
	norm_factor = 0
	i=x-khs
	while(i<=x+khs):
		j=y-khs
		while(j<=y+khs):
			gs = gauss(x-i,y-j,2.0)
			norm_factor += gs
			temp += gs*img[i][j]
 			j+=1
		i+=1
	s[x][y] = int(round(temp/norm_factor))

def gauss_smooth(I, window_size):
	smoothed_img = np.zeros(I.shape)
	i=window_size[0]/2
	while(i<len(I)-window_size[0]/2):
		j=window_size[1]/2
		while(j<len(I[0])-window_size[1]/2):
			smooth(I,smoothed_img,i,j,window_size)
			j+=1
		i+=1
	return smoothed_img

def non_maxima(Rp):
	candidates=[]
	for i in xrange(5,len(Rp)-5):
		for j in xrange(5,len(Rp[0])-5):
			max_flag = 0
			n = Rp[i][j]
			for k in range(-5,6):
				if (max_flag == 1):
					break
				for h in range(-5,6):
					if n < Rp[i+k][j+h]:
						max_flag = 1
						break

			if(max_flag == 0):
				candidates.append((i,j))
	return candidates

def HarrisCorner(img):
	k = 0.04
	Ix = gradient_x(img)
	Iy = gradient_y(img)
	##cv2.imwrite("gradient_x.pgm",Ix)
	##cv2.imwrite("gradient_y.pgm",Iy)

	A = gauss_smooth(Ix*Ix,(5,5))
	B = gauss_smooth(Iy*Iy,(5,5))
	C = gauss_smooth(Ix*Iy,(5,5))
	##cv2.imwrite("A.pgm",A)
	##cv2.imwrite("B.pgm",B)

	det = (A*B)-(C*C)
	trace = (A+B)
	Rp = det - (k * (trace * trace))

	candidate_list = non_maxima(Rp)

	corner_list=[]
	for c in candidate_list:
		if(Rp[c[0]][c[1]] > 100000):
			corner_list.append(cv2.KeyPoint(c[1],c[0],1))

	remove_list=[]
	i = 0
	for point in corner_list:
		j = 0
		for test_point in corner_list:
			if i != j:
				dist = (point.pt[0]-test_point.pt[0])**2+(point.pt[1]-test_point.pt[1])**2
				if dist<200:
					point_score = Rp[int(point.pt[1])][int(point.pt[0])]
					test_score  = Rp[int(test_point.pt[1])][int(test_point.pt[0])]
					if(point_score > test_score):
						if remove_list.count(j) == 0:
							remove_list.append(j)
					else:
						if remove_list.count(i) == 0:
							remove_list.append(i)
						break
			j = j+1
		i = i+1

	for point in sorted(remove_list, key=int, reverse=True):
		del corner_list[point]

	print len(corner_list)
	return corner_list

def remove_parallel_lines(lines):
	minThresh = 15
	vert = []
	hori = []
	newVert = []
	lastVert = []
	for line in lines:
		x1,y1,x2,y2,w = line
		angle = np.abs(np.rad2deg(np.arctan2(y1 - y2, x1 - x2)))
		if angle<105 and angle>75:
			vert.append(line)
		elif angle>165 or angle<15:
			 hori.append(line)

	i = 0
	for line in vert:
		l = np.sqrt((line[0]-line[2])**2+(line[1]-line[3])**2)
		j = 0
		for test in vert:
			if i != j and j > i:
				 d1 = np.sqrt((test[0]-line[0])**2+(test[1]-line[1])**2)
				 d2 = np.sqrt((test[0]-line[2])**2+(test[1]-line[3])**2)
				 d3 = np.sqrt((test[2]-line[0])**2+(test[3]-line[1])**2)
				 d4 = np.sqrt((test[2]-line[2])**2+(test[3]-line[3])**2)
				 minDist = sorted([d1,d2,d3,d4])[0]
				 if minDist < minThresh:
					 newVert.append([i,j])
			j = j+1
		i = i+1

	index = np.argmax(np.array(newVert))
	conVert = np.zeros((newVert[index/2][index%2]+1,newVert[index/2][index%2]+1),dtype=np.int)
	conVert = conVert-1
	for f,s in newVert:
		c = np.where(conVert == s)
		if np.array(c).size==0:
			cx = np.where(conVert == f)
			if np.array(cx).size==0:
				conVert[f,s] = s
			elif np.array(cx).size>0:
				conVert[cx[0][0],s] = s
		elif np.array(c).size>0:
			conVert[c[0][0],f] = f

	print conVert.shape[0]
	for i in xrange(conVert.shape[0]):
		c = np.where(conVert[i,:] != -1)
		y = []
		points = []
		a = 0
		if len(c[0])>0:
			for j in xrange(len(c[0])):
				index = c[0][j]
				points.append(vert[index])
				y.append(vert[index][1])
				y.append(vert[index][3])

			points.append(vert[i])
			y.append(vert[i][1])
			y.append(vert[i][3])
			miny = sorted(y)[0]
			maxy = sorted(y,reverse=True)[0]

			for p in points:
				if p[1] == miny or p[1] == maxy:
					if (a == 0):
						t1 = [int(p[0]),int(p[1])]
						a = a + 1
					elif (a == 1):
						t2 = [int(p[0]),int(p[1])]
						break
				if p[3] == miny or p[3] == maxy:
					if a == 0:
						t1 = [int(p[2]),int(p[3])]
						a = a + 1
					elif a == 1:
						t2 = [int(p[2]),int(p[3])]
						break

			lastVert.append([t1,t2])

	print len(lastVert)
	return lastVert


if __name__ == "__main__":

	image = cv2.imread("circuit3.jpg",0)
	image = imutils.resize(image,width=720)
	gray = image.copy()
	# draw found lines
	img_dst = cv2.imread("circuit3.jpg")
	img_dst = imutils.resize(img_dst,width=720)
	imgx = img_dst.copy()
	corner_list = HarrisCorner(image)
	lines = lsd(gray)
	for i in xrange(lines.shape[0]):
	    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
	    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
	    width = lines[i, 4]
	    cv2.line(imgx, pt1, pt2, (randint(0,255), randint(0,255),randint(0,255)), 1)
	lastVert = remove_parallel_lines(lines)

	for p1,p2 in lastVert:
		cv2.line(img_dst, (p1[0],p1[1]), (p2[0],p2[1]), (randint(0,255), randint(0,255), randint(0,255)),3 )


	cv2.drawKeypoints(img_dst, corner_list,img_dst)
	cv2.imshow('imgx',imgx)
	cv2.imshow('img',img_dst)
	cv2.waitKey(0)
