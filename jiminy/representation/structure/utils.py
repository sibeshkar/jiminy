import os
import cv2
import datetime
import numpy as np
# from jiminy.gym import Space

def getLabelForInput(inputObject, webdriver):
    name = inputObject.get_attribute("name")
    labelObject = None
    if name == "" or name is None:
        labelObject = webdriver.find_element_by_xpath("//input[@id='{}']//parent::label".format(inputObject.get_attribute("id")))
    else:
        labelObject = webdriver.find_element_by_xpath("//label[@for=({})]".format(name))
    return labelObject

def getInnerText(inputObject, webdriver):
    if inputObject.tag_name == "p" or inputObject.tag_name == "button" or inputObject.tag_name == "div":
        return inputObject.text
    elif inputObject.tag_name == "input" and inputObject.get_attribute('type') in ["radio", "checkbox"]:
        labelObject = getLabelForInput(inputObject, webdriver)
        return labelObject.text
    else:
        return ""

def getBoundingBoxCoords(objectInContext, webdriver):
    """
    :param objectInContext: the object for which we compute the bounding box
    """
    location = dict()
    location['x_1'] = objectInContext.location['x']
    location['y_1'] = objectInContext.location['y']
    location['x_2'] = objectInContext.location['x'] + objectInContext.size['width']
    location['y_2'] = objectInContext.location['y'] + objectInContext.size['height']
    if objectInContext.tag_name == "input" and objectInContext.get_attribute("type") in ["checkbox", "radio"]:
        objectLabel = getLabelForInput(objectInContext, webdriver)
        labelLocation = getBoundingBoxCoords(objectLabel, webdriver)
        location = combineLocations(location, labelLocation)
    return location

def combineLocations(location, labelLocation):
    location["x_1"] = min(location["x_1"], labelLocation["x_1"])
    location["y_1"] = min(location["y_1"], labelLocation["y_1"])
    location["x_2"] = max(location["x_2"], labelLocation["x_2"])
    location["y_2"] = max(location["y_2"], labelLocation["y_2"])
    return location

def saveScreenToFile(seleniumWebDriver):
    DIRNAME = os.getenv("JIMINY_LOGDIR")
    if DIRNAME is None:
        DIRNAME = "./"
    fileString = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f")
    saveString = "{}/{}.png".format(DIRNAME, fileString)
    seleniumWebDriver.get_screenshot_as_file(saveString)
    img = cv2.imread(saveString, 0)
    img = img[:300, :300]
    cv2.imwrite(saveString, img)
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
    tag_name = seleniumObject.tag_name
    type_name = seleniumObject.get_attribute("type")

    if tag_name == "button" or (tag_name == "input" and type_name == "text") or tag_name == "a":
        return "click"
    if tag_name == "input":
        if type_name == "radio":
            return type_name
        elif type_name == "checkbox":
            return type_name
        return "input"
    if tag_name == "p" or tag_name == "div":
        return "text"
    return "na"

def contains(ob, obj):
    if ob == obj:
        return False
    count = 0
    if ob.boundingBox["x_1"] <= obj.boundingBox["x_1"]:
        count+=1
    if ob.boundingBox["y_1"] <= obj.boundingBox["y_1"]:
        count+=1
    if ob.boundingBox["x_2"] >= obj.boundingBox["x_2"]:
        count+=1
    if ob.boundingBox["y_2"] >= obj.boundingBox["y_2"]:
        count+=1
    if count == 4:
        return True

def is_ancestor(obj, objectList):
    for ob in objectList:
        if contains(obj, ob):
            return True
    return False

def remove_ancestors(objectList):
    size_old = len(objectList) + 1
    size = len(objectList)
    while size < size_old:
        objectList_n = []
        for obj in objectList:
            if is_ancestor(obj, objectList):
                continue
            objectList_n.append(obj)
        objectList = objectList_n
        size, size_old = len(objectList), size
    objectList = list(set(objectList))
    return objectList
