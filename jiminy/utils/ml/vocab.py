"""
Vocabulary object for use with enttity <-> vector representations
"""

class Vocabulary(object):
    def __init__(self, objectlist):
        objectlist += ['NONE']
        self.key2sym = dict()
        self.sym2key = dict()
        for i, obj in enumerate(objectlist):
            self.key2sym[obj] = i
            self.sym2key[i] = obj
        self.length = len(objectlist)
        self.objectList = objectList

    def to_sym(self, entity_list):
        indices = np.zeros([len(entity_list)])
        for i, entity in enumerate(entity_list):
            if not entity in self.key2sym:
                indices[i] = self.key2sym['NONE']
            else:
               indices[i] = self.key2sym[entity]
        indices = indices.astype(np.int64)
        return indices

    def to_config(self):
        return json.dumps(self.objectList)

    def from_config(cls, config):
        objectList = json.loads(config)
        return cls(config)
