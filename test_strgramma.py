
# My lib
from model import dataset
import usurf
import strgramma
import tool
import copy

import os
from operator import itemgetter

segments = ["eyebrows", "eyes", "jaw", "mouth", "nose"]

algos = ["slope", "hog"]
wss = [3, 4, 5]
delta_bins = [15, 18, 20]
difs = [False]
edges = [True]

log = open("test_res_strgramma", "w")
best_recall = 0.0
best_parameter = [None, None, None, None, None]
try:
    for algo in algos:
        if algo=="hog":
            edges = [False, True]
        else:
            edges = [True]

        for ws in wss:
            for delta_bin in delta_bins:
                for dif in difs:
                    for edge in edges:
                        msg = "Test parameter with [ algo=%s, window size=%d, delta bin=%d, dif=%s, edge=%s ]" % (algo, ws, delta_bin, dif, edge)
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

                            strings = {}

                            for segment in segments:
                                im_draw = tool.imread('dataset/segment/'+segment+sid[-3:]+'.png', draw=False)
                                s = strgramma.extract(im_draw, ws=ws, algo=algo, dif=dif, edge=edge)
                                strings[segment] = s


                            data = {'id': sid, 'sex': sex, 'name':name, 'dtype': 'strgramma', 'strings': strings}
                            datas_raw.append(data)

                        f.close()

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

                            for i in range(1, 11):
                                strings = {}
                                for segment in segments:
                                    im = tool.imread("testset/"+str(i)+"/"+minisid+"-"+segment+".jpg")
                                    s = strgramma.extract(im, ws=ws, algo=algo, dif=dif, edge=edge)
                                    strings[segment] = s

                                datas = copy.deepcopy(datas_raw)

                                for data in datas:
                                    if data["dtype"]=="strgramma":
                                        data["point"] = 0
                                        for segment in segments:
                                            data["point"]+=strgramma.dist(strings[segment], data["strings"][segment])
                                    else:
                                        data["point"] = float("inf")

                                datas = sorted(datas, key=itemgetter('point'))

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

                        allrank_recall = (recall5+recall10+recall15)/3.0
                        if allrank_recall>best_recall:
                            best_recall = allrank_recall
                            best_parameter = [algo, ws, delta_bin, dif, edge]

except KeyboardInterrupt:
    pass

msg = "best mean all rank recall = %f parameter [ algo=%s, window size=%d, delta bin=%d, dif=%s, edge=%s ]" % (best_recall, best_parameter[0], best_parameter[1], best_parameter[2], best_parameter[3], best_parameter[4])
print msg
log.write(msg+"\n")
log.close()
