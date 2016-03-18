import usurf
import tool
import cv2
import numpy as np

im_draw = tool.imread('dataset/550610525.jpg', draw=True)

keypoints = usurf.detect(im_draw)

for keypoint in keypoints:
    cv2.circle(im_draw,keypoint["position"], int(keypoint["scale"]*10), (0,0,0), 1)

cv2.imwrite("res.jpg", im_draw)
cv2.waitKey(0)
