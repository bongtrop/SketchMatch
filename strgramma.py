import cv2
import tool
import math
import numpy as np

def gradient(im):
    h = im.shape[0]
    w = im.shape[1]

    dist = np.zeros((h-1,w-1), dtype=np.double)
    direct = np.zeros((h-1,w-1), dtype=np.double)

    nim = np.array(im, dtype=np.int64)

    for i in range(h-1):
        for j in range(w-1):
            dx = nim[i,j] - nim[i,j+1]
            dy = nim[i,j] - nim[i+1,j]

            dist[i,j] = math.sqrt(dx*dx + dy*dy)
            r = math.atan2(dy, dx)

            if r<0:
                r = 2*math.pi + r

            direct[i,j] = r

    return dist, direct


def hog(im, delta_bin):
    dist, direct = gradient(im)
    h = direct.shape[0]
    w = direct.shape[1]

    wbin = 360.0/delta_bin

    hist = np.zeros(delta_bin, dtype=np.double)

    for i in range(h):
        for j in range(w):
            delta = (direct[i, j]*180)/math.pi
            hist[int(delta/wbin)] += dist[i, j]

    return hist

def extract(im, ws=10, delta_bin=18, chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRETUVWXYZ"):
    h, w = im.shape
    res = ""

    wbin = 360.0/delta_bin

    for i in range(0, h, ws):
        for j in range(0, w, ws):
            window = im[i:i+ws, j:j+ws]
            hist = hog(window, delta_bin)
            if np.sum(hist)>0:
                res+=chars[np.argmin(hist)]

    return res

def dist(a, b):
    n, m = len(a), len(b)
    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

if __name__ == "__main__":
    dataset = ['550610487.jpg', '550610488.jpg', '550610525.jpg', '550610530.jpg']
    name = ['487', '488', '525', '530']

    f = open('res_test', 'w')

    f.write("sid\t")
    for n in name:
        f.write(n+"\t")

    f.write("\n\n")

    i = 0

    for d in dataset:
        f.write(name[i]+"\t")
        i+=1
        for dd in dataset:
            im1 = tool.imread('test/head/'+d, draw=False)
            im2 = tool.imread('test/input/'+dd, draw=True)

            s1 = extract(im1)
            s2 = extract(im2)

            s = str(dist(s1, s2))

            f.write(s+"\t")

        f.write("\n")

    f.close()
