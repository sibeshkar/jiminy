import os
import cv2
import datetime

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
    seleniumWebDriver.get_screenshot_as_file("{}/{}.png".format(DIRNAME, fileString))
    return fileString

def getPixelsFromFile(fname):
    img = cv2.imread(fname,'RGB')
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

def checkInputType(seleniumObject):
    """
    :param seleniumObject
    :returns boolean

    Checks if the element is an input type
    """
    if seleniumObject.tag_name() == "input":
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

