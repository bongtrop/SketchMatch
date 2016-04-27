from model import dataset
import tool
import usurf
import strgramma

segments = ["eyebrows", "eyes", "jaw", "mouth", "nose"]

f = open('dataset/data', 'r')

lines = f.readlines()

for line in lines:
    line = line.strip()
    words = line.split(' ')

    sid = words[0]
    sex = words[1]
    name = words[2]+" "+words[3]

    print "Process sid " + sid

    strings = {}

    for segment in segments:
        im_draw = tool.imread('dataset/segment/'+segment+sid[-3:]+'.png', draw=False)
        s = strgramma.extract(im_draw)
        strings[segment] = s


    data = {'id': sid, 'sex': sex, 'name':name, 'dtype': 'strgramma', 'strings': strings}
    dataset.add(data)

    print "Extract String Gramma "+ sid +" [Done]"

    im_draw = tool.imread('dataset/raw/'+sid+'.jpg', draw=True)

    keypoints = usurf.detect(im_draw)
    usurf.extract(im_draw, keypoints)

    data = {'id': sid, 'sex': sex, 'name':name, 'dtype': 'usurf', 'keypoints': keypoints}
    dataset.add(data)

    print "Extract usurf "+ sid +" [DONE]"
    print ""
