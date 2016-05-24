
# My lib
from model import dataset
import usurf
import strgramma
import tool

import os
from operator import itemgetter
import copy
import json

octaves = [4]
periods = [5]
scales = [1.2]
hessians = [0.4, 0.5]

f = open('testset/data', 'r')
log = open('test_res_usurf', 'w')

lines = f.readlines()
try:
    for octave in octaves:
        for period in periods:
            for scale in scales:
                for hessian in hessians:
                    msg = "Test parameter with [ octave=%d, period=%d, scale=%f, hessian=%f]" % (octave, period, scale, hessian)
                    print msg
                    log.write(msg+"\n")
                    f = open('dataset/data', 'r')

                    lines = f.readlines()
                    datas_raw = []

                    for line in lines:
                        line = line.strip()
                        words = line.split(' ')

                        sid = words[0]
                        sex = words[1]
                        name = words[2]+" "+words[3]

                        im_draw = tool.imread('dataset/raw/'+sid+'.jpg', draw=True)

                        keypoints = usurf.detect(im_draw, octave=octave, period=period, scale=scale, hessian=hessian)
                        usurf.extract(im_draw, keypoints)

                        data = {'id': sid, 'sex': sex, 'name':name, 'dtype': 'usurf', 'keypoints': keypoints}
                        datas_raw.append(data)

                    f.close()

                    with open('data_usurf.json', 'w') as outfile:
                        json.dump(datas_raw, outfile)

                    '''
                    with open('data_usurf.json', 'r') as infile:
                        datas_raw = json.load(infile)
                    '''


                    f = open('testset/data', 'r')

                    lines = f.readlines()
                    cc5 = 0.0
                    cc10 = 0.0
                    cc15 = 0.0
                    ccn5 = 0.0
                    ccn10 = 0.0
                    ccn15 = 0.0

                    for line in lines:
                        line = line.strip()
                        words = line.split(' ')

                        sid = words[0]
                        minisid = sid[-3:]
                        sex = words[1]
                        name = words[2]+" "+words[3]

                        c5 = 0
                        c10 = 0
                        c15 = 0
                        cn5 = 0
                        cn10 = 0
                        cn15 = 0

                        for i in range(1,11):
                            im = tool.imread("testset/"+str(i)+"/"+minisid+"-full.jpg")
                            keypoints = usurf.detect(im, octave=octave, period=period, scale=scale, hessian=hessian)
                            usurf.extract(im, keypoints)

                            lk = len(keypoints)

                            datas = copy.deepcopy(datas_raw)

                            for data in datas:
                                if data["dtype"]=="usurf":
                                    data["point"] = tool.match(keypoints, data["keypoints"])*1.0/lk
                                else:
                                    data["point"] = 0.0

                            datas = sorted(datas, key=itemgetter('point'), reverse=True)

                            rank = 1

                            for data in datas:
                                if data["id"]==sid:
                                    break

                                rank+=1

                            cn5+=5
                            cn10+=5
                            cn15+=5

                            if rank<=5:
                                c5+=1
                                cn5-=1

                            if rank<=10:
                                c10+=1
                                cn10-=1

                            if rank<=15:
                                c15+=1
                                cn15-=1

                        cc5+=c5/10.0
                        cc10+=c10/10.0
                        cc15+=c15/10.0

                        ccn5+=cn5/10.0/(len(lines)-1)
                        ccn10+=cn10/10.0/(len(lines)-1)
                        ccn15+=cn15/10.0/(len(lines)-1)

                    recall5 = cc5*1.0/len(lines)
                    recall10 = cc10*1.0/len(lines)
                    recall15 = cc15*1.0/len(lines)

                    precision5 = recall5/5.0
                    precision10 = recall10/10.0
                    precision15 = recall15/15.0

                    f5 = 0
                    f10 = 0
                    f15 = 0

                    if recall5+precision5>0:
                        f5 = 2.0*precision5*recall5/(recall5+precision5)

                    if recall10+precision10>0:
                        f10 = 2.0*precision10*recall10/(recall10+precision10)

                    if recall15+precision15>0:
                        f15 = 2.0*precision15*recall15/(recall15+precision15)

                    fallout5 = ccn5*1.0/len(lines)
                    fallout10 = ccn10*1.0/len(lines)
                    fallout15 = ccn15*1.0/len(lines)

                    msg = "Mean 5 rank [ recall=%.4f precision=%.4f f-measure=%.4f fallout=%.4f ]\n" % (recall5, precision5, f5, fallout5)
                    msg += "Mean 10 rank [ recall=%.4f precision=%.4f f-measure=%.4f fallout=%.4f ]\n" % (recall10, precision10, f10, fallout10)
                    msg += "Mean 15 rank [ recall=%.4f precision=%.4f f-measure=%.4f fallout=%.4f ]\n" % (recall15, precision15, f15, fallout15)
                    print msg
                    log.write(msg+"\n")

except KeyboardInterrupt:
    pass


log.close()
