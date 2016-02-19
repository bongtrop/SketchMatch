import numpy as np
from enum import Enum
from scipy import signal

W = 0.9

class Kernal(Enum):
	Dx = 1
	Dy = 2
	Dxy = 3

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def imintegral(im):
	res = np.zeros(im.shape, dtype=np.double)
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
	res = np.ones((h, w), dtype=np.double)*value

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
		l = left + (size - 2*w - 1)/2
		t = top + (size - 2*h - 1)/2

		return fastsum(imint, l, top+1, w, h) - fastsum(imint, l+w+1, t, w, h) - fastsum(imint, l, t+h+1, w, h) + fastsum(imint, l+w+1, t+h+1, w, h)


def convolute(imint, size=9, kernal=Kernal.Dx):
	h = imint.shape[0]
	w = imint.shape[1]
	skip = size/2

	res = np.zeros(imint.shape, dtype=np.double)

	for i in range(size/2, h-size/2):
		for j in range(size/2, w-size/2):
			res[i, j] = fastmask(imint, j-size/2, i-size/2, size, kernal)

	mx = np.max(res)
	mn = np.min(res)

	res = (res-mn)/(mx-mn)

	return res

def scale2size(scale):
	return int(scale/1.2 * 9)

def size2scale(size):
	return round(size/9.0 * 1.2 *10)/10

def discriminant(imint, size, w=0.9):
	dx = convolute(imint, size, Kernal.Dx)
	dy = convolute(imint, size, Kernal.Dy)
	dxy = convolute(imint, size, Kernal.Dxy)

	res = np.zeros(dx.shape, dtype=float)

	for i in range(dx.shape[0]):
		for j in range(dx.shape[1]):
			res[i, j] = dx[i, j] * dy[i, j] - (w*dxy[i, j])**2

	return res

def nonmaxima(octaves, th = 0.6):
	res = []
	for octave in octaves:
		for ii in range(len(octave)):
			period = octave[ii]
			im = period["im"]

			for i in range(im.shape[0]):
				for j in range(im.shape[1]):
					if im[i,j]>th:
						top = max(0, i-1)
						bottom = min(im.shape[0],i+2)
						left = max(0, j-1)
						right = min(im.shape[1],j+2)

						m = 0
						if ii>0:
							pev = octave[ii-1]["im"]
							m = np.max(pev[top:bottom, left:right])

						if ii<len(octave)-1:
							nxt = octave[ii+1]["im"]
							m = max(m, np.max(nxt[top:bottom, left:right]))

						m = max(m, np.max(im[top:bottom, left:right]))

						if im[i,j]==m:
							position = (j, i)
							scale = period["scale"]
							res.append({"position": position, "scale": scale})

	return res

def detect(im, octave=3, period=4, scale=1.2, hessian=0.6):
	imint = imintegral(np.array(im))
	s = scale2size(scale)
	add = 6
	octaves = []

	for i in range(octave):

		periods = []
		ss = s

		for j in range(period):
			imres = discriminant(imint, ss)
			periods.append({"scale": size2scale(s), "im":imres})
			ss+=add

		octaves.append(periods)
		s+=add
		add*=2

	return nonmaxima(octaves, hessian)

def extract(im, keypoints):
	for keypoint in keypoints:
		position = keypoint["position"]
		scale = keypoint["scale"]

		s = 20*scale
		x = position[0]
		y = position[1]

		window = im[y-s/2:y+s/2, x-s/2:x+s/2]

		ks = int(2*scale)
		kdx = np.ones((ks, ks), dtype=np.double)
		kdx[:,:ks/2] = -1
		kdy = np.ones((ks, ks), dtype=np.double)
		kdy[:ks/2,:] = -1

		des = []
		ws = s/4

		for i in range(4):
			for j in range(4):
				subwindow = window[i*ws:(i+1)*ws, j*ws:(j+1)*ws]
				dx = signal.convolve2d(subwindow, kdx, mode='valid')
				dy = signal.convolve2d(subwindow, kdy, mode='valid')

				des+=[np.sum(dx), np.sum(dy), np.sum(np.abs(dx)), np.sum(np.abs(dy))]

		keypoint["des"] = des
