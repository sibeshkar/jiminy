#! /usr/bin/env python

# See README.txt for information and build instructions.

from __future__ import print_function
import demo_pb2
import sys

filepath = 'recordings/recording_1563174043/proto.rbs'

class VNCDemonstration(object):
    def __init__(self, filepath):
        self.demonstration = demo_pb2.Demonstration()
        self.pixel_format = dict()
        self.initialize_from_file(filepath)
        self.initMsg()
        print(filepath)

    def initialize_from_file(self, filepath):
        with open(filepath, "rb") as f:
            self.demonstration.ParseFromString(f.read())

    def initMsg(self):
        self.fbheight = self.demonstration.initmsg.FBHeight
        self.fbwidth = self.demonstration.initmsg.FBWidth
        self.rfbversion = self.demonstration.initmsg.RfbVersion
        self.rfbheader = self.demonstration.initmsg.RfbHeader
        self.start_time = self.demonstration.initmsg.StartTime
        self.desktopname = self.demonstration.initmsg.DesktopName
        self.pixel_format = {
            "bpp" : self.demonstration.initmsg.PixelFormat.BPP,
            "depth" : self.demonstration.initmsg.PixelFormat.Depth,
            "bigendian" : self.demonstration.initmsg.PixelFormat.BigEndian,
            "truecolor" : self.demonstration.initmsg.PixelFormat.TrueColor,
            "redmax" : self.demonstration.initmsg.PixelFormat.RedMax,
            "greenmax" : self.demonstration.initmsg.PixelFormat.GreenMax,
            "bluemax" : self.demonstration.initmsg.PixelFormat.BlueMax,
            "redshift" : self.demonstration.initmsg.PixelFormat.RedShift,
            "greenshift" : self.demonstration.initmsg.PixelFormat.GreenShift,
            "blueshift" : self.demonstration.initmsg.PixelFormat.BlueShift,
        }
    def listPointerEvents(self):
        for pointerevent in self.demonstration.pointerevents:
            print(pointerevent.Mask, pointerevent.X, pointerevent.Y, pointerevent.timestamp)
    def listFBUpdates(self):
        for fbupdate in self.demonstration.fbupdates:
            print(len(fbupdate.rectangles), fbupdate.timestamp)

    def printInitMsg(self):
        print("FBHeight:", self.fbheight)
        print("FBWidth:", self.fbwidth)
        print("RfbVersion:", self.rfbversion)
        print("RfbHeader:", self.rfbheader)
        print("StartTime:", self.start_time)
        print("DesktopName:", self.desktopname)
        print("PixelFormat", self.pixel_format)

class FBUpdateReader(object):
    def __init__(self, filepath):
        self.demonstration = demo_pb2.Demonstration()
        self.initialize_from_file(filepath)
        self.idx = 0
        self.max_length = len(self.demonstration.fbupdates)

    def initialize_from_file(self, filepath):
        with open(filepath, "rb") as f:
            self.demonstration.ParseFromString(f.read())
       # self.fbupdates = [x for x in self.demonstration.fbupdates]

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx > self.max_length:
            raise StopIteration
        else:
            self.idx +=1
            return self.demonstration.fbupdates[self.idx] 
        
class KeyEventReader(object):
    def __init__(self, filepath):
        self.demonstration = demo_pb2.Demonstration()
        self.initialize_from_file(filepath)
        self.idx = 0
        self.max_length = len(self.demonstration.keyevents)

    def initialize_from_file(self, filepath):
        with open(filepath, "rb") as f:
            self.demonstration.ParseFromString(f.read())
       # self.fbupdates = [x for x in self.demonstration.fbupdates]

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx > self.max_length:
            raise StopIteration
        else:
            self.idx +=1
            return self.demonstration.keyevents[self.idx] 

class PointerEventReader(object):
    def __init__(self, filepath):
        self.demonstration = demo_pb2.Demonstration()
        self.initialize_from_file(filepath)
        self.idx = 0
        self.max_length = len(self.demonstration.pointerevents)

    def initialize_from_file(self, filepath):
        with open(filepath, "rb") as f:
            self.demonstration.ParseFromString(f.read())
       # self.fbupdates = [x for x in self.demonstration.fbupdates]

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx > self.max_length:
            raise StopIteration
        else:
            self.idx +=1
            return self.demonstration.pointerevents[self.idx] 






fbr = iter(PointerEventReader(filepath))
i = 0

while True:
    try:
        fbu = next(fbr)
        print("PointerEvent number:", i)
        i += 1
    except IndexError:
        break
    print("X: %d, Y: %d, Mask: %d" % (fbu.X, fbu.Y, fbu.Mask))

# Main procedure:  Reads the entire address book from a file and prints all
#   the information inside.
# if len(sys.argv) != 2:
#    print("Usage:", sys.argv[0], "DEMO_FILE")
#    sys.exit(-1)
# vncdemo = VNCDemonstration(sys.argv[1])
# #vncdemo.printInitMsg()
# #vncdemo.listPointerEvents()
# vncdemo.listFBUpdates()
