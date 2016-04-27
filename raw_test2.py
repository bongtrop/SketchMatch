
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
wrs = [2, 3, 4, 5, 6]
delta_bins = [10, 12, 15, 18, 20]
difs = [False, True]
edges = [False, True]

log = open("test_res", "w")
best_recall = 0.0
best_parameter = [None, None, None, None, None]
try:
    for algo in algos:
        if algo=="hog":
            edges = [False, True]
        else:
            edges = [True]

        for wr in wrs:
            for delta_bin in delta_bins:
                for dif in difs:
                    for edge in edges:
                        msg = "Test parameter with [ algo=%s, window ratio=%d, delta bin=%d, dif=%s, edge=%s ]" % (algo, wr, delta_bin, dif, edge)
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
                                s = strgramma.extract(im_draw, wr=wr, algo=algo, dif=dif, edge=edge)
                                strings[segment] = s


                            data = {'id': sid, 'sex': sex, 'name':name, 'dtype': 'strgramma', 'strings': strings}
                            datas_raw.append(data)

                        f.close()

                        f = open('testset/data', 'r')

                        lines = f.readlines()
                        cc = 0

                        for line in lines:
                            line = line.strip()
                            words = line.split(' ')

                            sid = words[0]
                            minisid = sid[-3:]
                            sex = words[1]
                            name = words[2]+" "+words[3]

                            strings = {}
                            for segment in segments:
                                im = tool.imread("testset/"+minisid+"-"+segment+".jpg")
                                s = strgramma.extract(im, wr=wr, algo=algo, dif=dif, edge=edge)
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

                            str_rank = 1

                            for data in datas:
                                if data["id"]==sid:
                                    break

                                str_rank+=1

                            if str_rank<=5:
                                cc+=1

                        recall = cc*1.0/len(lines)
                        msg = "Mean recall = " + str(cc*1.0/len(lines))
                        print msg
                        log.write(msg+"\n")

                        if recall>best_recall:
                            best_recall = recall
                            best_parameter = [algo, wr, delta_bin, dif, edge]

except KeyboardInterrupt:
    pass

print "best recall = %f parameter [ algo=%s, window ratio=%d, delta bin=%d, dif=%s, edge=%s ]" % (best_recall, best_parameter[0], best_parameter[1], best_parameter[2], best_parameter[3], best_parameter[4])
