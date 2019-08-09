"""
Vocabulary object for use with enttity <-> vector representations
"""
import numpy as np
import json
import tensorflow as tf

class Vocabulary(object):
    def __init__(self, objectlist, pretrained=None):
        self.pretrained = pretrained
        if self.pretrained is None:
            self.pretrained = dict()
        objectlist += ['NONE']
        self.key2sym = dict()
        self.sym2key = dict()
        for i, obj in enumerate(objectlist):
            self.key2sym[obj] = i
            self.sym2key[i] = obj
        self.length = len(objectlist)
        self.objectList = objectlist

    def to_sym(self, entity_list):
        indices = np.zeros([len(entity_list)])
        for i, entity in enumerate(entity_list):
            if not entity in self.key2sym:
                indices[i] = self.key2sym['NONE']
            else:
               indices[i] = self.key2sym[entity]
        indices = indices.astype(np.int64)
        return indices

    def to_key(self, entity_list):
        values = [None for _ in entity_list]
        for i, entity in enumerate(entity_list):
            values[i] = self.sym2key[entity]
        return values

    def to_config(self):
        return json.dumps(self.objectList)

    def from_config(cls, config):
        objectList = json.loads(config)
        return cls(objectList)

    def get_embedding_initializer(self, shape=(None, 50)):
        if self.pretrained is None:
            print("Using random embeddings")
            return tf.keras.initializer.RandomUniform(minval=-1.0, maxval=1.0)
        self.pretrained['NONE'] = np.random.uniform(-1., 1., size=shape[-1])
        embedding_matrix = np.array([self.pretrained[obj] for obj in self.objectList])
        class EmbeddingInit(tf.keras.initializers.Initializer):
            def __init__(self, embedding_matrix):
                super(EmbeddingInit, self).__init__()
                self.embedding_matrix = embedding_matrix

            def __call__(self, shape, dtype=None, partition_info=None):
                assert self.embedding_matrix.shape == shape, "Shape mismatch: {} expected : {}".format(self.embedding_matrix.shape, shape)
                try :
                    tensor = tf.convert_to_tensor(self.embedding_matrix, dtype=dtype)
                except:
                    assert False, "Could not convert shape from {} to required {}".format(self.embedding_matrix.dtype, dtype)
                return tensor
        return EmbeddingInit(embedding_matrix)
