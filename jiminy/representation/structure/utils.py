import os
import cv2
import datetime
import numpy as np
from jiminy.gym import Space

def getBoundingBoxCoords(objectInContext):
    """
    :param objectInContext: the object for which we compute the bounding box
    """
    location = dict()
    location['x_1'] = objectInContext.location['x']
    location['y_1'] = objectInContext.location['y']
    location['x_2'] = objectInContext.location['x'] + objectInContext.size['width']
    location['y_2'] = objectInContext.location['y'] + objectInContext.size['height']
    return location

def saveScreenToFile(seleniumWebDriver):
    DIRNAME = os.getenv("DATADUMPDIR")
    if DIRNAME is None:
        DIRNAME = "./"
    fileString = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    saveString = "{}/{}.png".format(DIRNAME, fileString)
    seleniumWebDriver.get_screenshot_as_file(saveString)
    return saveString

def getPixelsFromFile(fname):
    img = cv2.imread(fname)
    return np.asarray(img, dtype=np.float32)

def getActionList(seleniumObject):
    """
    Returns the possible ways in which we can interact with an object
    """
    actionList = []
    if checkClickableObject(seleniumObject):
        actionList.append('left-click')
    if checkTypableObject(seleniumObject):
        actionList.append('type')
    return actionList

def checkClickableObject(seleniumObject):
    """
    TODO(prannayk): write this
    """
    return True

def checkTypableObject(seleniumObject):
    """
    TODO(prannayk): write this
    """
    return True

def getPixelsForBoundingBox(betadom, boundingBox):
    if boundingBox["x_2"] < boundingBox["x_1"]:
        boundingBox["x_1"], boundingBox["x_2"] = boundingBox["x_2"], boundingBox["x_1"]
    if boundingBox["y_2"] < boundingBox["y_1"]:
        boundingBox["y_1"], boundingBox["y_2"] = boundingBox["y_2"], boundingBox["y_1"]
    assert(boundingBox["x_1"] >= 0 and boundingBox["x_1"] <= betadom.pixels.shape[0])
    assert(boundingBox["x_2"] >= 0 and boundingBox["x_2"] <= betadom.pixels.shape[0])
    assert(boundingBox["y_1"] >= 0 and boundingBox["y_1"] <= betadom.pixels.shape[0])
    assert(boundingBox["x_2"] >= 0 and boundingBox["x_2"] <= betadom.pixels.shape[0])
    height = int(boundingBox["x_2"] - boundingBox["x_1"])
    width = int(boundingBox["y_2"] - boundingBox["y_1"])
    pixels = np.ndarray([height, width, 3], dtype=np.float32)
    pixels = betadom.pixels[int(boundingBox["x_1"]):int(boundingBox["x_2"]), int(boundingBox["y_1"]):int(boundingBox["y_2"]),:]
    return pixels

def checkInputType(seleniumObject):
    """
    :param seleniumObject
    :returns boolean

    Checks if the element is an input type
    """
    if seleniumObject.tag_name == "input":
        return True

def checkTextInputType(seleniumObject):
    """
    :param seleniumObject
    :returns boolean

    Helps us differentiate between form input which require clicking and require typing
    """
    if 'text' in seleniumObject.get_attribute('type'):
        return True

def getObjectType(seleniumObject):
    """
    Returns the type of object in jiminy
    """
    objectTypeList = []
    if checkInputType(seleniumObject):
        objectTypeList.append('input')
        if checkTextInputType(seleniumObject):
            objectTypeList.append('text-input')
        else:
            objectTypeList.append('form-input')


def ListActionSpace(Space):
    def __init__(self, action_space_list):
        self.action_space_list = action_space_list

    def sample(self):
        size = len(self.action_space_list)
        return self.action_space_list[np.random.randint(low=0, high=size)].sample()

    def contains(self, action):
        if np.array([action.contains(action) for action in self.action_space_list]).any():
            return True
        return False


def flatten(actionlist):
    return ListActionSpace(actionlist)
