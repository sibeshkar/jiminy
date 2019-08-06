from jiminy.gym import ObservationWrapper
from jiminy.representation.structure import utils
from jiminy.global_protos import DomObjectInstance
import json
import datetime
from google.protobuf import json_format

class JiminyBaseInstancePb2(ObservationWrapper):
    def __init__(self, json_string=None):
        self.pb2 = DomObjectInstance()
        if not json_string is None:
            self._observation(json_string)

    def _observation(self, json_string):
        self.queryv = None
        if isinstance(json_string, dict):
            json_string = json_string["dom"]
        if json_string is None:
            return None
        json_format.Parse(json_string, self.pb2)
        return self

    @property
    def query(self):
        if self.queryv is not None:
            return self.queryv.content
        for dom in self.pb2.objects:
            if dom.type == "query":
                self.queryv = dom
                return dom.content

    @property
    def objects(self):
        return self.pb2.objects

class JiminyBaseObject(object):
    def __init__(self, betaDOM, seleniumObject=None, seleniumDriver=None,
            boundingBox=None, objectType=None,
            referenceTag=None, innerText=None, value=None, focused=None):
        self.metadata = dict()
        if seleniumObject != None:
            self.boundingBox = utils.getBoundingBoxCoords(seleniumObject, seleniumDriver)
            self.objectType = utils.getObjectType(seleniumObject)
            self.focused = (seleniumObject == seleniumDriver.switch_to.active_element)
            self.value = seleniumObject.get_attribute('value')
            self.innerText = utils.getInnerText(seleniumObject, seleniumDriver)
            self.metadata["TrueTag"] = seleniumObject.tag_name
            self.metadata["inferTime"] = datetime.datetime.now().strftime("%H%M%S")
        else:
            if boundingBox == None:
                raise ValueError("Bounding box can not be None")
            if objectType == None:
                raise ValueError("Object Type can not be none")
            self.boundingBox = boundingBox
            self.objectType = objectType
        self.objectPixels = utils.getPixelsForBoundingBox(betaDOM, self.boundingBox)
        self.children = []

    def __str__(self):
        jiminyDict = dict()
        jiminyDict['boundingBox'] = self.boundingBox
        jiminyDict['objectType'] = self.objectType
        jiminyDict['focused'] = self.focused
        jiminyDict['value'] = self.value
        jiminyDict['innerText'] = self.innerText
        # jiminyDict['metadata'] = self.getMetadata()
        result = json.dumps(jiminyDict)
        return result

    def appendMetadata(self, tupleKV):
        if len(tupleKV) != 2:
            raise ValueError
        self.metadata[tupleKV[0]] = tupleKV[1]

    def getMetadata(self):
        import json
        return json.dumps(self.metadata)
