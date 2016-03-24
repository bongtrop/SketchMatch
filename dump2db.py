from model import dataset
import tool
import usurf
import strgramma

f = open('dataset/data', 'r')

lines = f.readlines()

for line in lines:
    line = line.strip()
    words = line.split(' ')

    sid = words[0]
    sex = words[1]
    name = words[2]+" "+words[3]

    print "Process sid " + sid

    im_draw = tool.imread('dataset/'+sid+'.jpg', draw=True)
    s = strgramma.extract(im_draw)

    data = {'id': sid, 'sex': sex, 'name':name, 'dtype': 'strgramma', 'string': s}
    dataset.add(data)
    print "Extract String Gramma [Done]"
    '''
    im_raw = tool.imread('dataset/'+sid+'.jpg')

    keypoints = usurf.detect(im_raw)
    usurf.extract(im_raw, keypoints)

    data = {'id': sid, 'sex': sex, 'name':name, 'dtype': 'usurf_raw', 'keypoints': keypoints}
    dataset.add(data)

    print "usurf raw "+ sid +" [DONE]"

    im_draw = tool.imread('dataset/'+sid+'.jpg', draw=True)

    keypoints = usurf.detect(im_draw)
    usurf.extract(im_draw, keypoints)

    data = {'id': sid, 'sex': sex, 'name':name, 'dtype': 'usurf_draw', 'keypoints': keypoints}
    dataset.add(data)

    print "usurf draw "+ sid +" [DONE]"
    print ""
    '''
