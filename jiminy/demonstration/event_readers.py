from __future__ import print_function
from jiminy.demonstration.demo_pb2 import Demonstration
import sys
import heapq
import jiminy
import jiminy.demonstration
from jiminy import utils
from jiminy import spaces
from jiminy import utils
from jiminy.vncdriver import fbs_reader, vnc_proxy_server, vnc_client
from jiminy import spaces as vnc_spaces
from PIL import Image
import io
import cv2
import numpy as np
import json
import shlex
from subprocess import Popen, PIPE
import os



class DemoReader(object):
    def __init__(self,filepath):
        """
        DemoReader is inner most iterator, takes in a .rbs filepath and returns an Iterator Object that returns processed
        batches like this:

        demo = iter(DemoReader("demo_423412132312.rbs"))
        obs, actions, rewards, dones, doms = next(demo)
        """
        self.demonstration = Demonstration()
        self.initialize_from_file(filepath)
        self.idx = 0
        self.max_length = len(self.demonstration.batches)
    def initialize_from_file(self, filepath):
        with open(filepath, "rb") as f:
            self.demonstration.ParseFromString(f.read())
    def __iter__(self):
        return self

    def process(self, batch):
        processed_batch = dict()
        for iterator in batch.iterators:
            if iterator.type == 'obs':
                img = Image.open(io.BytesIO(iterator.events[0].event))
                opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                processed_batch['obs'] = opencvImage
            elif iterator.type == 'actions':
                # processed_batch[iterator.type] = iterator.events
                processed_batch['actions'] = self.process_actions(self.process_json(iterator.events[0]))
            elif iterator.type == 'records':
                rewards, dones, doms = self.process_records(self.process_json(iterator.events[0]))
                processed_batch['doms'] = doms
                processed_batch['rewards'] = rewards
                processed_batch['dones'] = dones
        # img = Image.open(io.BytesIO(processed_batch['obs'][0].event))
        # opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # processed_batch['obs'] = opencvImage
        return processed_batch

    def process_json(self, events):
        bytes_array = events.event
        data_json = bytes_array.decode('utf8').replace("'", '"')
        data = json.loads(data_json)
        #data_list = json.dumps(data, indent=4, sort_keys=True)
        return data

    def process_records(self, events):
        dom = []
        reward = []
        done = []

        for event in events:
            if len(event['Body']) !=0:
                if "Obs" in event['Body']:
                    dom.append(event['Body']['Obs'])
                else:
                    dom.append("{}")
                if "Reward" in event['Body']:
                    reward.append(event['Body']['Reward'])
                else:
                    reward.append(0.0)
                if "Done" in event['Body']:
                    done.append(event['Body']['Done'])
                else:
                    done.append(False)
        return reward, done, dom

    def process_actions(self, events):
        actions = []
        for event in events:
            if event is not None:
                if event['Mask'] is not None:
                    event_p = self.pointer_event(event['X'], event['Y'], event['Mask'])
                    actions.append(event_p)
                else:
                    event_p = self.key_event(event['Key'], event['Down'])
        return actions

    def key_event(self, key, down):
        event = spaces.KeyEvent(key, down)
        return event

    def pointer_event(self, x, y, buttonmask):
        event = spaces.PointerEvent(x, y, buttonmask)
        return event

    def __next__(self):
        if self.idx > self.max_length:
            raise StopIteration
        else:
            self.idx +=1
            data_list = self.process(self.demonstration.batches[self.idx-1])
            return data_list['obs'],data_list['actions'],data_list['rewards'], data_list['dones'], data_list['doms']
    
filename = "demo_1565953935.rbs"

class VNCDemonstration(object):

    """
    VNCDemonstration is a helpful wrapper around DemoReader in order to collate a folderful of .rbs files (of the form: server.rbs, client.rbs, record.rbs)
    into a single .rbs file that DemoReader can process. 
        demo = iter(DemoReader("demo_423412132312.rbs"))
        obs, actions, rewards, dones, doms = next(demo)
    """
    def __init__(self, demo_file=None):
        self.binary_path()
        self.idx = 0
        if demo_file is not None:
            self.initialized = True
            self.reader = iter(DemoReader(os.path.abspath(demo_file)))

    def create_demo(self, directory, fps=10, speedup=3.0):
        """
        Takes in a folderful of recordings of different events and collates all
        into a single .rbs file using the library: github.com/sibeshkar/demoparser
        """
        directory = os.path.abspath(directory)
        cmd = self.bin_path + " -fps=" + str(fps) +  " -speedup=" + str(speedup) + " -directory=" + directory
        code, out , err = get_exitcode_stdout_stderr(cmd)
        print(code, out, err)
        self.demo_file=out.decode("utf-8").split('\n')[2]
        self.initialized = True
        self.reader = iter(DemoReader(self.demo_file))

    def binary_path(self):
        from sys import platform
        path = jiminy.demonstration.__file__[:-11]
        if platform == "linux" or platform == "linux2":
            self.bin_path = path + "bin/demoparser_linux"
        elif platform == "darwin":
            self.bin_path = path + "bin/demoparser_darwin"
        elif platform == "win32":
            print("Windows Binary not available for demo processing")

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx > self.max_length:
            raise StopIteration
        else:
            return next(self.reader)

    def play(self):
        if self.initialized:
            while True:
                obs, actions, rewards, dones, doms = next(self.reader)
                cv2.imshow('image',obs)
                print("-"*20)
                print("The Record is {}".format(actions))
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()

def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)

    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    #
    return exitcode, out, err

# demo = VNCDemonstration()
# demo.create_demo("recording_1565930432", 20, 3.0)
# demo.play()
