from jiminy.sdk.wrappers.core import Transformation

class DynamicTrigger(Transformation):
    def with_trigger(self, keyName, checkFn):
        if isinstance(keyName, str):
            self.keyName = [keyName]
            self.checkFn = [checkFn]
        else:
            self.keyName = keyName
            self.checkFn = checkFn
        return self

    def _update_trigger(self, inputNode):
        for key in self.keyName:
            assert key in inputNode.value, "Can not find required key for trigger: {} in inputNode: {}".format(key, inputNode)
        for idx, key in enumerate(self.keyName):
            if not self.checkFn[idx](self.checkinputNode.value[key]):
                return False
        return True
