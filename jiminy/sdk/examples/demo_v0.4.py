import pyautogui as pg
import cv2
import subprocess
from pynput import keyboard
from jiminy.sdk.transformations import PixelToSelectedText
from jiminy.sdk.blocks import GoogleKeepRobot
import time
import numpy as np
import argparse

argparser = argparse.ArgumentParser("Demo-v0.4")
argparser.add_argument("--theme", dest="theme", type=str,
        help="Theme of your OS X", default="light")

args = argparser.parse_args()

COMBINATIONS = [
        (keyboard.Key.alt, keyboard.Key.f1),
        ]

current = set()

def filterText(text):
    pieces = text.strip().split()
    pieces = [''.join(filter(lambda x : x.isalnum(), piece)) for piece in pieces]
    return ' '.join(pieces)

def execute():
    img = pg.screenshot('screenshot.png').convert('RGB')
    output = subprocess.check_output(["osascript", "-e", 'tell application "Google Chrome" to get bounds of the window 1']).decode().strip().split(',')
    xmin, ymin, xmax, ymax = [int(x) for x in output]
    img = np.array(img)[ymin*2:ymax*2, xmin*2:xmax*2, ::-1]
    inputs = {
            "img" : img
            }
    t = time.time()
    pix2text = PixelToSelectedText()
    text = pix2text.forward(inputs)
    gkr = GoogleKeepRobot(chrome=(xmin, ymin, xmax, ymax), theme=args.theme, keep_theme="light")
    gkr.forward({
        "img" : img,
        "selected-text" : filterText(text['selected-text']),
        "title" : "Paul Graham - Economic Inequality"
        })

def clicked(key):
    if any([key in COMBO for COMBO in COMBINATIONS]):
        current.add(key)
        if any(all([k in frozenset(current) for k in COMBO]) for COMBO in COMBINATIONS):
            print("Executing")
            execute()

def on_release(key):
    if any([key in COMBO for COMBO in COMBINATIONS]):
        current.remove(key)

if __name__ == "__main__":
    with keyboard.Listener(on_press=clicked, on_release=on_release) as listener:
        listener.join()
