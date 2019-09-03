import xxhash
import time
import numpy as np

baseConfiguration = dict()

class BaseGraphEntity(object):
    def __init__(self, name, input_dict={}, output_dict={}):
        assert name is not None and name != "", "Expected name to be something meaningful, found: {}".format(name)
        self.name = name
        self.input_dict = input_dict
        self.output_dict = output_dict
        self.cache = dict()
        self.configuration = None

    def forward(self, inputs):
        assert inputs is not None and isinstance(inputs, dict), "inputs should be a dict: {}".format(inputs)
        if self.input_dict is not None:
            for key in self.input_dict:
                assert key in inputs, "Can not find key: {} required for input in: {}".format(key, inputs)
        return self._forward(inputs)

    def _forward(self, inputs):
        raise NotImplementedError

    def initializer(self, name="default"):
        if hasattr(self, 'configuration') and self.configuration is not None:
            self._save_to_configuration(self.configuration)
        if not name in self.cache:
            self.configuration = self.cache[name] = self._initializer(name)
        else:
            self.configuration = self.cache[name]
        self.load_from_configuration(self.cache[name])
        return self.configuration

    def _initializer(self, name="default"):
        return {}

    def load_from_configuration(self, configuration):
        assert configuration is not None, "configuraton to load an object from can not be null"
        return self._load_from_configuration(configuration)

    def _load_from_configuration(self, configuration):
        pass

    def _save_to_configuration(self, configuration):
        pass

    @property
    def shape(self):
        return self.output_dict

    @property
    def input(self):
        return self.input_dict

class BaseGraph(object):
    def __init__(self, name=""):
        self.nodes = dict()
        self.edges_incoming = dict()
        self.edges_outgoing = dict()
        self.transformations = dict()
        self.name = name
        self.input_dict = dict()
        self.output_dict = dict()

    def as_default(self):
        class BaseGraphScopeHandler(object):
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
        return BaseGraphScopeHandler(self)

    def add_node(self, node):
        assert isinstance(node, Block), "Node: {} is not of type Block".format(node)
        assert not node.name in self.nodes, "Node by name {} is already present in the graph".format(node.name)
        self.nodes[node.name] = node
        self.edges_incoming[node.name] = dict()
        self.edges_outgoing[node.name] = dict()

    def add_edge(self, n_in, n_out, transformation):
        self.transformations[transformation.name] = transformation
        self.edges_outgoing[n_in.name][n_out.name] = transformation.name
        self.edges_incoming[n_out.name][n_in.name] = transformation.name

    def __str__(self):
        return "{}\n".format(self.name) + str({
            "nodes" : [str(self.nodes[node]) for node in self.nodes],
            "edges_incoming" : self.edges_incoming,
            "edges_outgoing" : self.edges_outgoing,
            "transformations" : [str(self.transformations[transformation_name]) for transformation_name in self.transformations]
            })


class Block(BaseGraphEntity):
    def __init__(self, name="", input_dict={}, output_dict={}):
        super(Block, self).__init__(name + str(xxhash.xxh32(str(time.time())).intdigest()),
                input_dict=input_dict,
                output_dict=output_dict)
        baseConfiguration["graph"].add_node(self)

    def __str__(self):
        return "name: {}, output: {}".format(self.name, self.output_dict)

class Transformation(BaseGraphEntity):
    def __init__(self, name="", input_dict={}, output_dict={}):
        super(Transformation, self).__init__(name,
                input_dict=input_dict,
                output_dict=output_dict)

    @property
    def value(self):
        assert False, "Transformation objects do not have persistent values, call the transformation with a block"

    def __call__(self, source, target):
        if self.input_dict is not None:
            assert source.output_dict == self.input_dict, "Type mismatch: {} and {} in input to Transformation: {}".format(source.output_dict, self.input_dict, self.name)
        if self.output_dict is not None:
            assert np.array([key in target.input_dict for key in self.output_dict]).all(), "Type mismatch: {} and {} in output to Transformation: {}".format(target.input_dict, self.output_dict, self.name)

        baseConfiguration["graph"].add_edge(source, target, self)

    def __str__(self):
        return "name: {}, input: {}, output: {}".format(self.name, self.input_dict, self.output_dict)

baseConfiguration["graph"] = BaseGraph("default_graph")

if __name__ == "__main__":
    graph = BaseGraph("not_default")
    with graph.as_default() as current_graph:
        block = Block(input_dict={"img" : [288, 288, 4]}, output_dict={"img" : [144, 144, 3]})
        betadom = Block(input_dict={"betadom" : [10,10]}, output_dict={"betadom" : [None, 10]})
        transformation = Transformation(name="img-to-betadom", input_dict={"img" : [144, 144, 3]}, output_dict={"betadom" : [10, 10]})(source=block, target=betadom)
        print(current_graph)
    print(baseConfiguration["graph"])
