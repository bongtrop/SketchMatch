import cv2
import tool
import math

def extract(im, ws=5, delta_bin=18, chars="abcdefghijklmnopqrstuvwxyz"):
    e = cv2.Canny(im,100,200)
    h, w = e.shape
    res = ""

    for i in range(0, h, ws):
        for j in range(0, w, ws):
            window = e[i:i+ws, j:j+ws]

            if window.shape[0]!=ws or window.shape[1]!=ws:
                continue

            window[1:ws-1, 1:ws-1] = 0

            p = []

            for ii in range(0, ws):
                for jj in range(0, ws):
                    if window[ii, jj]>0:
                        p.append((ii, jj))

            if len(p)>1:
                start = p[0]
                end = p[-1]

                delta = math.atan2(end[0] - start[0], end[1]- start[1])
                delta = abs((delta*180)/math.pi)
                res+=chars[int(delta/delta_bin)]

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


dataset = ['550610487.jpg', '550610488.jpg', '550610525.jpg', '550610530.jpg']

f = open('res_test', 'w')

for d in dataset:
    for dd in dataset:
        im1 = tool.imread('test/head/'+d, draw=False)
        im2 = tool.imread('test/input/'+dd, draw=True)

        s1 = extract(im1)
        s2 = extract(im2)

        s = d+" "+dd+" "+str(dist(s1, s2))

        f.write(s+"\r\n")
