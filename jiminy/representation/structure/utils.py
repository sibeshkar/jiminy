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
