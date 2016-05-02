# -*- coding: utf-8 -*-

'''
Author: Pongsakorn Sommalai (bongtrop@gmail.com)
Date: 25/11/2015
Description: Server blur beer photo
'''

# Default
import json
import cgi
import random
import time
from operator import itemgetter

# My lib
from model import dataset
import usurf
import strgramma
import tool

# Web Server
from twisted.web.server import Site
from twisted.web.resource import Resource
from twisted.internet import reactor

segments = ["eyebrows", "eyes", "jaw", "mouth", "nose"]

def gen_random_name(l):
   return ''.join([random.choice('0123456789ABCDEF') for x in range(l)])

class IdentiFace(Resource):
    isLeaf = True

    def render_GET(self, request):
        request.responseHeaders.addRawHeader(b"content-type", b"application/json")
        skip = 0
        limit = 0
        if request.args.get("skip"):
            skip = int(request.args.get("skip")[0])

        if request.args.get("limit"):
            limit = min(100, int(request.args["limit"][0]))


        datas = dataset.get(filter={'dtype': 'usurf_raw'}, skip=skip, limit=limit)

        res = []

        for data in datas:
            res.append({'id': data['id'], 'name': data['name'], 'sex': data['sex']})

        return json.dumps({"status": "success", "data": res})

    def render_POST(self, request):
        self.headers = request.getAllHeaders()

        img = cgi.FieldStorage(
            fp = request.content,
            headers = self.headers,
            environ = {'REQUEST_METHOD':'POST', 'CONTENT_TYPE': self.headers['content-type']})

        full = ""
        jaw = ""
        eyebrows = ""
        eyes = ""
        nose = ""
        mouth = ""

        if img.has_key("full"):
            filename = gen_random_name(32)+'.jpg'
            out = open('logs/'+filename, 'wb')
            out.write(img["full"].value)
            out.close()
            full = filename

        if img.has_key("jaw"):
            filename = gen_random_name(32)+'.jpg'
            out = open('logs/'+filename, 'wb')
            out.write(img["jaw"].value)
            out.close()
            jaw = filename

        if img.has_key("eyebrows"):
            filename = gen_random_name(32)+'.jpg'
            out = open('logs/'+filename, 'wb')
            out.write(img["eyebrows"].value)
            out.close()
            eyebrows = filename

        if img.has_key("eyes"):
            filename = gen_random_name(32)+'.jpg'
            out = open('logs/'+filename, 'wb')
            out.write(img["eyes"].value)
            out.close()
            eyes = filename

        if img.has_key("nose"):
            filename = gen_random_name(32)+'.jpg'
            out = open('logs/'+filename, 'wb')
            out.write(img["nose"].value)
            out.close()
            nose = filename

        if img.has_key("mouth"):
            filename = gen_random_name(32)+'.jpg'
            out = open('logs/'+filename, 'wb')
            out.write(img["mouth"].value)
            out.close()
            mouth = filename

        if img.has_key("sex") and img.has_key("algo"):

            sex = img["sex"].value
            datas = dataset.get(filter={'sex': sex})

            if img["algo"].value=="usurf" and full!="":
                im = tool.imread('logs/'+full)
                keypoints = usurf.detect(im)
                usurf.extract(im, keypoints)

                lk = len(keypoints)

                for data in datas:
                    if data["dtype"]=="usurf":
                        data["point"] = tool.match(keypoints, data["keypoints"])*1.0/lk
                    else:
                        data["point"] = 0.0

            elif img["algo"].value=="strgramma" and jaw!="" and eyebrows!="" and eyes!="" and nose!="" and mouth!="":
                strings = {}
                strings["jaw"] = strgramma.extract(tool.imread('logs/'+jaw))
                strings["eyebrows"] = strgramma.extract(tool.imread('logs/'+eyebrows))
                strings["eyes"] = strgramma.extract(tool.imread('logs/'+eyes))
                strings["nose"] = strgramma.extract(tool.imread('logs/'+nose))
                strings["mouth"] = strgramma.extract(tool.imread('logs/'+mouth))

                for data in datas:
                    if data["dtype"]=="strgramma":
                        data["point"] = 0.0
                        for segment in segments:
                            data["point"]+=strgramma.dist(strings[segment], data["strings"][segment])

                        data["point"] = 1.0/data["point"]
                    else:
                        data["point"] = 0.0
            else:
                return json.dumps({"status": "fail", "detail": "Wrong Parameter"})

            datas = sorted(datas, key=itemgetter('point'), reverse=True)

            res = []

            for i in range(min(10, len(datas))):
                data = datas[i]
                res.append({'id': data['id'], 'name': data['name'], 'sex': data['sex'], 'point': float(data['point'])})

            request.responseHeaders.addRawHeader(b"content-type", b"application/json")
            return json.dumps({"status": "success", "data": res})
        else:
            return json.dumps({"status": "fail", "detail": "Wrong Parameter"})

class Image(Resource):
    def __init__(self, iid):
        Resource.__init__(self)
        self.id = iid

    def render_GET(self, request):
        request.setHeader("content-type", "image/jpeg")
        try:
            f = open("dataset/raw/"+self.id+".jpg", "rb")
            return f.read()
        except Exception as error:
            return ""

class Main(Resource):
    def getChild(self, name, request):
        if name.lower()=="identiface":
            return IdentiFace()

        return Image(name)


resource = Main()
factory = Site(resource)
reactor.listenTCP(8080, factory)
reactor.run()
