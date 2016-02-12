import cv2
import numpy as np
from enum import Enum

W = 0.9

class Kernal(Enum):
	Dx = 1
	Dy = 2
	Dxy = 3

def imintegral(im):
	res = np.zeros(im.shape, dtype=np.int32)
	h = im.shape[0]
	w = im.shape[1]

	for i in range(h):
		for j in range(w):
			if i==0 and j==0:
				res[i, j] = im[i, j]
			elif i==0:
				res[i, j] = im[i, j] + res[i, j-1]
			elif j==0:
				res[i, j] = im[i, j] + res[i-1, j]
			else:
				res[i, j] = im[i, j] + res[i-1, j] + res[i, j-1] - res[i-1, j-1]

	return res

def fastsum(imint, l, t, w, h):

	bottom = t+h-1
	right = l+w-1
	left = l
	top = t

	if left==0 and top==0:
		return imint[bottom, right]
	elif top==0:
		return imint[bottom, right] - imint[bottom, left-1]
	elif left==0:
		return imint[bottom, right] - imint[top-1, right]
	else:
		return imint[bottom, right] - imint[top-1, right] - imint[bottom, left-1] + imint[top-1, left-1]

def impad(im, left=0, top=0, right=0, bottom=0, value=0):
	h = im.shape[0] + top + bottom
	w = im.shape[1] + left + right
	res = np.ones((h, w), dtype=im.dtype)*value

	res[top:h-bottom, left:w-right] = im
	return res

def fastmask(imint, left, top, size, kernal):
	if kernal==Kernal.Dy:
		h = size/3
		w = h*2 - 1
		l = left + (size - w)/2
		return fastsum(imint, l, top, w, h) - 2*fastsum(imint, l, top+h, w, h) + fastsum(imint, l, top+2*h, w, h)

	elif kernal==Kernal.Dx:
		w = size/3
		h = w*2 - 1
		t = top + (size - h)/2
		return fastsum(imint, left, t, w, h) - 2*fastsum(imint, left+w, t, w, h) + fastsum(imint, left+2*w, t, w, h)

	else:
		w = size/3
		h = w
		return fastsum(imint, left+1, top+1, w, h) - fastsum(imint, left+w+2, top+1, w, h) - fastsum(imint, left+1, top+h+2, w, h) + fastsum(imint, left+w+2, top+h+2, w, h)

def convolute(imint, size=9, kernal=Kernal.Dx):
	h = imint.shape[0]
	w = imint.shape[1]
	skip = size/2

	res = np.zeros(imint.shape, dtype=imint.dtype)

	for i in range(h-size):
		for j in range(w-size):
			res[i, j] = fastmask(imint, i, j, size, kernal)

	return res

def scale2size(scale):
	return int(scale/1.2 * 9)

def size2scale(size):
	return round(size/9.0 * 1.2 *10)/10

def discriminant(imint, size, w=0.9):
	dx = convolute(imint, size, Kernal.Dx)
	dy = convolute(imint, size, Kernal.Dy)
	dxy = convolute(imint, size, Kernal.Dxy)

	res = np.array(dx.shape, dtype=float)

	for i in range(dx.shape[0]):
		for j in range(dx.shape[1]):
			res[i, j] = dx[i, j] * dy[i, j] - (w*dxy[i, j])**2

	return res
