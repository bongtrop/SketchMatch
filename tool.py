from PIL import Image, ImageFilter, ImageOps
import numpy as np
import itertools
import operator
import math

def dodge(a, b, alpha):
    return min(int(a*255/(256-b*alpha)), 255)

def im2draw(im1, blur=13, alpha=1.0):
    im2 = im1.copy()
    im2 = ImageOps.invert(im2)

    im2 = im2.filter(ImageFilter.GaussianBlur(radius=blur))

    width, height = im1.size
    for x in range(width):
        for y in range(height):
            a = im1.getpixel((x, y))
            b = im2.getpixel((x, y))
            im1.putpixel((x, y), dodge(a, b, alpha))

    return np.asarray(im1)

def imread(filename, draw=False):
    im = Image.open(filename).convert("L")

    if draw:
        im = im2draw(im)

    return np.asarray(im)

def match(kps_a, kps_b, ratio = 0.6):
    num=0
    for kp_a in kps_a:
        mn1 = float("inf")
        mn2 = float("inf")
        for kp_b in kps_b:
            dps=sum(itertools.starmap(operator.mul,zip(kp_a["des"], kp_b["des"])))
            delta = math.acos(max(min(dps,1), -1))

            if delta < mn1:
                mn2 = mn1
                mn1 = delta
            elif delta < mn2:
                mn2 = delta

        if mn1 < mn2*ratio:
            num += 1

    return num
