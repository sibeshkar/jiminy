#! /usr/bin/env python

# See README.txt for information and build instructions.

from __future__ import print_function
import demo_pb2
import sys

class VNCDemonstration(object):
    def __init__(self, filepath):
        self.demonstration = demo_pb2.Demonstration()
        self.pixel_format = dict()
        self.initialize_from_file(filepath)
        self.initMsg()

    def initialize_from_file(self, filepath):
        with open(sys.argv[1], "rb") as f:
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

    def printInitMsg(self):
        print("FBHeight:", self.fbheight)
        print("FBWidth:", self.fbwidth)
        print("RfbVersion:", self.rfbversion)
        print("RfbHeader:", self.rfbheader)
        print("StartTime:", self.start_time)
        print("DesktopName:", self.desktopname)
        print("PixelFormat", self.pixel_format)



# Main procedure:  Reads the entire address book from a file and prints all
#   the information inside.
if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "DEMO_FILE")
  sys.exit(-1)

vncdemo = VNCDemonstration(sys.argv[1])

vncdemo.printInitMsg()
