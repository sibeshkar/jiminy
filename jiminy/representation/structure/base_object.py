from jiminy.representation.structure import utils
import json

class JiminyBaseObject(object):
    def __init__(self, betaDom, seleniumObject=None, boundingBox=None, objectType=None, actionList=None):
        if seleniumObject != None:
            self.boundingBox = utils.getBoundingBoxCoords(seleniumObject)
            self.objectType = utils.getObjectType(seleniumObject)
            self.actionList = utils.getActionList(seleniumObject)
        else:
            if boundingBox == None:
                raise ValueError("Bounding box can not be None")
            if objectType == None:
                raise ValueError("Object Type can not be none")
            if actionList == None:
                raise ValueError("ActionList can not be null")
            self.boundingBox = boundingBox
            self.objectType = objectType
            self.actionList = actionList
        self.objectPixels = utils.getPixelsForBoundingBox(betaDom, self.boundingBox)
        n_inv = 1. / len(betaDom.getActionableStateList())
        # initialize that all actionable states to be equally likely
        self.softmaxActionableState = dict(map(lambda x : (x, n_inv), betaDom.getActionableStateList()))
        self.children = []
        self.metadata = dict({
            "ObjectType" : "default"
            })

    def toString(self):
        jiminyDict = dict()
        jiminyDict['boundingBox'] = self.boundingBox
        jiminyDict['objectType'] = self.objectType
        jiminyDict['actionList'] = self.actionList
        result = json.dumps(jiminyDict)
        return result

    def appendMetadata(self, tupleKV):
        if len(tupleKV) != 2:
            raise ValueError
        self.metadata[tupleKV[0]] = tupleKV[1]

    def getMetadata(self):
        import json
        return json.dumps(self.metadata)

    def getActionList(self):
        return utils.flatten([action.action_space for self.softmaxActionableState.values()])
