
# My lib
from model import dataset
import usurf
import strgramma
import tool

import os
from operator import itemgetter

f = open("res_usurf", "w")

datas = dataset.get()

for (dirpath, dirnames, filenames) in os.walk("testset"):
    for filename in filenames:
        names = filename.split("-")
        if len(names)==2 and names[1]=="full.jpg":
            sid = "550610"+names[0]

            print "[*] Test sid "+sid

            im = tool.imread('testset/'+filename)
            keypoints = usurf.detect(im)
            usurf.extract(im, keypoints)

            lk = len(keypoints)

            for data in datas:
                if data["dtype"]=="usurf_draw" or data["dtype"]=="usurf_raw":
                    data["point"] = tool.match(keypoints, data["keypoints"])*1.0/lk
                else:
                    data["point"] = 0.0

            datas = sorted(datas, key=itemgetter('point'), reverse=True)

            rank = 1
            for data in datas:
                if data["id"]==sid:
                    break

                if data["dtype"]=="usurf_draw" or data["dtype"]=="usurf_raw":
                    rank+=1

            print "[+] rank " + str(rank)

            f.write(sid+" "+str(rank)+"\n")

f.close()
