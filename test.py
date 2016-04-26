
# My lib
from model import dataset
import usurf
import strgramma
import tool

import os
from operator import itemgetter

segments = ["eyebrows", "eyes", "jaw", "mouth", "nose"]

f = open('testset/data', 'r')
log = open('res', 'w')

lines = f.readlines()

for line in lines:
    line = line.strip()
    words = line.split(' ')

    sid = words[0]
    minisid = sid[-3:]
    sex = words[1]
    name = words[2]+" "+words[3]

    print "Process sid " + sid
    print "[+] USURF test"
    im = tool.imread("testset/"+minisid+"-full.jpg")
    keypoints = usurf.detect(im)
    usurf.extract(im, keypoints)

    lk = len(keypoints)

    datas = dataset.get()

    for data in datas:
        if data["dtype"]=="usurf":
            data["point"] = tool.match(keypoints, data["keypoints"])*1.0/lk
        else:
            data["point"] = 0.0

    datas = sorted(datas, key=itemgetter('point'), reverse=True)

    usurf_rank = 1

    for data in datas:
        if data["id"]==sid:
            break

        usurf_rank+=1

    print "Rank is "+str(usurf_rank)

    print "[+] String Gramma test"
    strings = {}
    for segment in segments:
        im = tool.imread("testset/"+minisid+"-"+segment+".jpg")
        s = strgramma.extract(im)
        strings[segment] = s

    datas = dataset.get()

    for data in datas:
        if data["dtype"]=="strgramma":
            data["point"] = 0
            for segment in segments:
                data["point"]+=strgramma.dist(strings[segment], data["strings"][segment])
        else:
            data["point"] = float("inf")

    datas = sorted(datas, key=itemgetter('point'))

    str_rank = 1

    for data in datas:
        if data["id"]==sid:
            break

        str_rank+=1

    print "Rank is "+str(str_rank)
    print ""

    log.write(sid+" "+str(usurf_rank)+" "+str(str_rank)+"\n")

log.close()
