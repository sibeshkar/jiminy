import xxhash
import time

baseConfiguration = dict()

class BaseGraphEntity(object):
    def __init__(self, name):
        self.name = name
        assert name is not None and name != "", "Expected name to be something meaningful, found: {}".format(name)

    def forward(self, inputs):
        return self._forward(inputs)

    def _forward(self, inputs):
        raise NotImplementedError

class BaseGraph(object):
    def __init__(self, name=""):
        self.nodes = dict()
        self.edges = dict()
        self.transformations = dict()
        self.name = name

    def as_default(self):
        class DefaultBaseGraph(object):
            def __init__(self, graph):
                self.graph = graph

            def __enter__(self):
                self.prev_graph = baseConfiguration["graph"]
                baseConfiguration["graph"] = self.graph
                return self.graph

            def __exit__(self, type, value, traceback):
                if type is not None:
                    return
                assert hasattr(self, 'prev_graph') and self.prev_graph is not None, "A graph was not set while entering scope."
                baseConfiguration["graph"] = self.prev_graph
                return True
        return DefaultBaseGraph(self)

    def add_node(self, node):
        assert isinstance(node, Block), "Node: {} is not of type Block".format(node)
        assert not node.name in self.nodes, "Node by name {} is already present in the graph".format(node.name)
        self.nodes[node.name] = node
        self.edges[node.name] = dict()

    def add_edge(self, n1, n2, transformation):
        self.transformations[transformation.name] = transformation
        self.edges[n1.name][n2.name] = transformation.name
        self.edges[n2.name][n1.name] = transformation.name

    def __str__(self):
        return "{}\n".format(self.name) + str({
            "nodes" : [str(self.nodes[node]) for node in self.nodes],
            "edges" : self.edges,
            "transformations" : self.transformations
            })

    def _reset(self):
        [self.transformations[transformation].reset() for transformation in self.transformations]
        self.ready = {node : False for self.nodes}

    def forward_node(self, name, inputs):
        return self._forward_node(name, inputs)

    def _forward_node(self, name, inputs):
        raise NotImplementedError


class Block(BaseGraphEntity):
    def __init__(self, name="", output_dict={}):
        super(Block, self).__init__(name + str(xxhash.xxh32(str(time.time())).intdigest()))
        self.output_dict = output_dict
        baseConfiguration["graph"].add_node(self)

    def __str__(self):
        return "name: {}, output: {}".format(self.name, self.output_dict)

    @property
    def shape(self):
        return self.output_dict

class Transformation(BaseGraphEntity):
    def __init__(self, name="", input_dict={}, output_dict={}):
        assert input_dict is not None, "Expected input_dict to not be None"
        assert output_dict is not None, "Expected output_dict to not be None"
        super(Transformation, self).__init__(name)
        self.input_dict = input_dict
        self.output_dict = output_dict

    def call(self, input_value):
        raise NotImplementedError

    def __call__(self, input_value):
        for key in self.input_dict:
            assert (not key in input_value), "Key {} not found in {}".format(key, input_value)
            assert self.input_dict[key].shape == input_value[key].shape, "Shape mismatch {}, expected: {} found: {}".format(key, self.input_dict[key].shape, input_value[key].shape)
        self.call(input_value)

    @property
    def shape(self):
        return self.output_dict

    @property
    def input(self):
        return self.input_dict

baseConfiguration["graph"] = BaseGraph("default_graph")

if __name__ == "__main__":
    graph = BaseGraph("not_default")
    with graph.as_default() as current_graph:
        block = Block(output_dict={"img" : [144, 144, 3]})
        print(current_graph)

    print(baseConfiguration["graph"])
