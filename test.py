import cv2
import strgramma

im = cv2.imread("testset/1/493-jaw.jpg", 0)
print strgramma.extract(im, ws=20, delta_bin=20, edge=True, edge_parameter=[200, 400])
