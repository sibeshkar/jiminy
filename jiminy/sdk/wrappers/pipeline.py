from jiminy.sdk.wrappers.core import BaseGraphNode, BaseGraph

class Pipeline(BaseGraphNode, BaseGraph):
    def __init__(self, name=""):
        super(BaseGraphEntity, self).__init__(name)
        super(BaseGraph, self).__init__(name)
        self.input_nodes = dict()
        self.output_nodes = dict()
        self.action_nodes = dict()

    def _add_input_node(self, node):
        self.input_nodes[node.name] = node
    def _add_output_node(self, node):
        self.output_nodes[node.name] = node
    def _add_action_node(self, node):
        self.action_nodes[node.name] = name

    def _forward(self, inputs):
        assert isinstance(inputs, dict), "Expcted inputs to pipeline to be a dictionary, found: {}".format(inputs)
        for key in self.input_nodes:
            assert key in inputs, "Need input for node: {}, not found in: {}".format(key, self.input_nodes)
            self.forward_node(key, inputs[key])

    def _forward_node(self, name, inputs):
        self.nodes[name].forward(inputs)
