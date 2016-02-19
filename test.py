import usurf
import cv2

filename = "550610505.jpg";

im = cv2.imread("photo/photo/"+filename, 0)
imres = cv2.imread("photo/photo/"+filename)

#sim = cv2.imread("photo/sketch/sketch_"+filename, 0)
#simres = cv2.imread("photo/sketch/sketch_"+filename)

listim = im.tolist()
#listsim = sim.tolist()

keypoints = usurf.detect(listim, 3, 4, 1.2, 0.4)
usurf.extract(im, keypoints)

print keypoints[0]
#skeypoints = usurf.detect(listsim, 3, 4, 1.2, 0.4)

for keypoint in keypoints:
	print str(keypoint["position"]) + " " + str(keypoint["scale"])
	cv2.circle(imres, keypoint["position"], int(keypoint["scale"]*20)/2, (0,0,255), 1)
	cv2.circle(imres, keypoint["position"], 2, (0,255,0), -1)

#for skeypoint in skeypoints:
#	print str(skeypoint["position"]) + " " + str(skeypoint["scale"])
#	cv2.circle(simres, skeypoint["position"], int(skeypoint["scale"]*20)/2, (0,0,255), 1)
#	cv2.circle(simres, skeypoint["position"], 2, (0,255,0), -1)

cv2.imshow("raw", imres)
#cv2.imshow("sketch", simres)
cv2.waitKey(0)
