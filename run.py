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
import detect

# Web Server
from twisted.web.server import Site
from twisted.web.resource import Resource
from twisted.internet import reactor

def gen_random_name(l):
   return ''.join([random.choice('0123456789ABCDEF') for x in range(l)])

class PoliceEye(Resource):
    isLeaf = True

    def render_GET(self, request):


    def render_POST(self, request):


resource = PoliceEye()
factory = Site(resource)
reactor.listenTCP(12345, factory)
reactor.run()
