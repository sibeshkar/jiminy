from jiminy.sdk.wrappers.core import *
from jiminy.sdk.wrappers.session import *

class Pipeline(object):
    def __init__(self, name=""):
        self.name = name
        self.input_nodes = dict()
        self.output_nodes = dict()
        self.action_nodes = dict()
        self.graph = BaseGraph(self.name + "-Graph")

    def _add_input_node(self, node):
        self.input_nodes[node.name] = node
        self.graph.add_node(node)
    def _add_output_node(self, node):
        self.output_nodes[node.name] = node
        self.graph.add_node(node)
    def _add_action_node(self, node):
        self.action_nodes[node.name] = name
        self.graph.add_node(node)
    def _add_edge(self, *args, **kwargs):
        self.graph.add_edge(*args, **kwargs)

    def __entry__(self):
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            return
        if self.prev_pipeline is not None:
            baseConfiguration["pipeline"] = self.prev_pipeline

    def as_default(self):
        self.prev_pipeline = baseConfiguration["pipeline"]
        baseConfiguration["pipeline"] = self
        return self

    def step(self, observation):
        output_nodes = list(self.output_nodes.keys())
        input_nodes = list(self.input_nodes.keys())
        if not hasattr(self, 'session'):
            self.session = Session(self.name + "-Session", self.graph)
        with self.session.as_default() as sess:
            result = sess.run(input_nodes=input_nodes, output_nodes=output_nodes, observation=observation)

        return result["values"], result["actions"], result["action_log_prob"]

    def probe(self, observation):
        output_nodes = list(self.output_nodes.keys())
        input_nodes = list(self.input_nodes.keys())
        if not hasattr(self, 'session'):
            self.session = Session(self.name + "-Session", self.graph)
        with self.session.as_default() as sess:
            result = sess.run(input_nodes=input_nodes, output_nodes=action_nodes, observation=observation)

        return result

baseConfiguration["pipeline"] = Pipeline(name="default-pipeline")
