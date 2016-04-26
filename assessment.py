
f = open("res", "r")

lines = f.readlines()

rank_th = 5

usurf_precision = 0
strgramma_precision = 0

usurf_recall = 0
strgramma_recall = 0

usurf_fallout = 0
strgramma_fallout = 0

usurf_f = 0
strgramma_f = 0

ldata = len(lines)

usurf_precision_s = 0
strgramma_precision_s = 0

usurf_recall_s = 0
strgramma_recall_s = 0

usurf_fallout_s = 0
strgramma_fallout_s = 0

usurf_f_s = 0
strgramma_f_s = 0

for line in lines:
    line = line.replace("\n", "")
    words = line.split(" ")

    sid = words[0]
    usurf_rank = words[1]
    str_rank = words[2]

    if int(usurf_rank)<=rank_th:
        usurf_precision = 1.0/rank_th
        usurf_recall = 1.0
        usurf_fallout = 4.0/ldata
        usurf_f = 2.0*usurf_precision*usurf_recall/(usurf_precision+usurf_recall)
    else:
        usurf_precision = 0
        usurf_recall = 0
        usurf_fallout = 5.0/ldata
        usurf_f = 0

    if int(str_rank)<=rank_th:
        strgramma_precision = 1.0/rank_th
        strgramma_recall = 1.0
        strgramma_fallout = 4.0/ldata
        strgramma_f = 2.0*strgramma_precision*strgramma_recall/(strgramma_precision+strgramma_recall)
    else:
        strgramma_precision = 0
        strgramma_recall = 0
        strgramma_fallout = 5.0/ldata
        strgramma_f = 0

    print sid+" "+str(usurf_precision)+" "+str(usurf_recall)+" "+str(usurf_fallout)+" "+str(usurf_f)+" | "+str(strgramma_precision)+" "+str(strgramma_recall)+" "+str(strgramma_fallout)+" "+str(strgramma_f)

    usurf_precision_s += usurf_precision;
    usurf_recall_s += usurf_recall
    usurf_fallout_s += usurf_fallout
    usurf_f_s += usurf_f

    strgramma_precision_s += strgramma_precision
    strgramma_recall_s += strgramma_recall
    strgramma_fallout_s += strgramma_fallout
    strgramma_f_s += strgramma_f


print "USURF mean "+ str(usurf_precision_s*1.0/len(lines)) + " "+ str(usurf_recall_s*1.0/len(lines)) + " "+ str(usurf_fallout_s*1.0/len(lines)) + " "+ str(usurf_f_s*1.0/len(lines))
print "String Gramma mean  "+ str(strgramma_precision_s*1.0/len(lines)) + " "+ str(strgramma_recall_s*1.0/len(lines)) + " "+ str(strgramma_fallout_s*1.0/len(lines)) + " "+ str(strgramma_f_s*1.0/len(lines))
