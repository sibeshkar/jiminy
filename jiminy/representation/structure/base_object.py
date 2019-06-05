from jiminy.representation.structure import utils
import json

class JiminyBaseObject(object):
    def __init__(self, betaDOM, seleniumObject=None, seleniumDriver=None,
            boundingBox=None, objectType=None,
            referenceTag=None, innerText=None, value=None, focused=None):
        if seleniumObject != None:
            self.boundingBox = utils.getBoundingBoxCoords(seleniumObject, seleniumDriver)
            self.objectType = utils.getObjectType(seleniumObject)
            self.focused = (seleniumObject == seleniumDriver.switch_to.active_element)
            self.value = seleniumObject.get_attribute('value')
            self.innerText = utils.getInnerText(seleniumObject, seleniumDriver)
        else:
            if boundingBox == None:
                raise ValueError("Bounding box can not be None")
            if objectType == None:
                raise ValueError("Object Type can not be none")
            if actionList == None:
                raise ValueError("ActionList can not be null")
            self.boundingBox = boundingBox
            self.objectType = objectType
        self.objectPixels = utils.getPixelsForBoundingBox(betaDOM, self.boundingBox)
        self.children = []
        self.metadata = dict({
            "ObjectType" : self.objectType
            })

    def toString(self):
        jiminyDict = dict()
        jiminyDict['boundingBox'] = self.boundingBox
        jiminyDict['objectType'] = self.objectType
        result = json.dumps(jiminyDict)
        return result

    def appendMetadata(self, tupleKV):
        if len(tupleKV) != 2:
            raise ValueError
        self.metadata[tupleKV[0]] = tupleKV[1]

    def getMetadata(self):
        import json
        return json.dumps(self.metadata)
