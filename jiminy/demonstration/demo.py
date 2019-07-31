#! /usr/bin/env python

# See README.txt for information and build instructions.

from __future__ import print_function
import demo_pb2
import sys
import heapq
from jiminy import utils
from jiminy import spaces
from jiminy import utils
from jiminy.vncdriver import fbs_reader, vnc_proxy_server, vnc_client
from jiminy import spaces as vnc_spaces

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
            return {
                "timestamp" : self.demonstration.fbupdates[self.idx].timestamp,
                "fbupdate" : self.demonstration.fbupdates[self.idx],
            } 
        
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
            return {
                "timestamp" : self.demonstration.keyevents[self.idx].timestamp,
                "keyevent" : self.demonstration.keyevents[self.idx],
            }
        
         

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
            return {
                "timestamp" : self.demonstration.pointerevents[self.idx].timestamp,
                "pointerevent" : self.demonstration.pointerevents[self.idx],
            }


class MergedEventReader(object):
    def __init__(self, *timestamped_iterables):
        """
        Args:
            timestamped_iterables: A set of iterables that return dictionaries. Each dictionary must contain a 'timestamp' key that is monotonically increasing
        """
        self.merged_iterables = heapq.merge(*timestamped_iterables, key=self._extract_timestamp)
        self.timestamp = None

    @staticmethod
    def _extract_timestamp(line):
        if not isinstance(line, dict):
            raise InvalidEventLogfileError("MultiIterator received a line that's not a dictionary: {}".format(line))
        try:
            return float(line['timestamp'])
        except KeyError:
            raise InvalidEventLogfileError("MultiIterator received a line without a timestamp: {}".format(line))

    
    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next item from our iterable lists, sorted by timestamp
        """
        data = next(self.merged_iterables)
        timestamp = data['timestamp']
        if self.timestamp and (timestamp < self.timestamp):
            raise InvalidEventLogfileError("MultiIterator received a line with an out of order timestamp: {}".format(data))

        self.timestamp = timestamp
        return data

class FBSEventReader(object):
    def __init__(self, filepath, paint_cursor=False):
        self.paint_cursor = paint_cursor
        observation_reader = FBUpdateReader(filepath)
        self.error_buffer = utils.ErrorBuffer()
        action_reader = PointerEventReader(filepath)
        self.merged_reader = MergedEventReader(observation_reader, action_reader)
        self.pixel_format = []
        self.observation_processor = vnc_client.VNCClient()


    


mer = MergedEventReader(PointerEventReader(filepath),FBUpdateReader(filepath))

# while True:
#     try:
#         fbu = next(mer)
#     except IndexError:
#         break
#     print(fbu)
# Main procedure:  Reads the entire address book from a file and prints all
#   the information inside.
# if len(sys.argv) != 2:
#    print("Usage:", sys.argv[0], "DEMO_FILE")
#    sys.exit(-1)
# vncdemo = VNCDemonstration(sys.argv[1])
# #vncdemo.printInitMsg()
# #vncdemo.listPointerEvents()
# vncdemo.listFBUpdates()
