from jiminy.sdk.wrappers.core import *
from jiminy.sdk.utilities import shapeMatcher
from jiminy.sdk.wrappers.trigger import TriggerTransformation
import queue

class DynamicGraph(object):
    def __init__(self, graph):
        self.name = "Dynamic" + graph.name
        self.nodes = {node : DynamicBlock(graph.nodes[node]) for node in graph.nodes}
        self.edges_incoming = graph.edges_incoming
        self.edges_outgoing = graph.edges_outgoing
        self.transformations = {transformation : DynamicTransformation(graph.transformations[transformation]) for transformation in graph.transformations}
        self.ready_queue = queue.Queue()

    def reset(self, session):
        [self.nodes[node].reset(session) for node in self.nodes]
        [self.transformations[t].reset(session) for t in self.transformations]

    def init(self, session):
        [self.nodes[node].init(session) for node in self.nodes]
        [self.transformations[t].init(session) for t in self.transformations]

    def forward_node(self, input_values, node_name):
        assert node_name in self.nodes, "node_name: {} not found in dynamic graph: {}".format(input_values, self)
        for node,transformation_n in self.edges_incoming[node_name]:
            transformation = self.transformations[transformation_n]
            output_values = transformation(self.nodes[node])
            for key in output_values:
                input_values[key] = output_values[key]

        self.nodes[node_name].update(input_values)
        self.set_triggers(node_name)
        for node,ts in self.edges_outgoing[node_name]:
            transformation = self.transformations[ts]
            if isinstance(transformation, DynamicTriggerTransformation) and not transformation.is_executable:
                continue
            self.nodes[node]._set_ready(node_name)
            if self.nodes[node].is_ready:
                self.ready_queue.put(node)

    def set_triggers(self, node_name):
        node = self.nodes[node_name]
        for _, ts in self.edges_outgoing[node_name]:
            self.transformations[ts].update_trigger(node.value)

    def search_smallest_graph(self, output_nodes):
        q = queue.Queue()
        for node in output_nodes : q.put(node)
        marked = dict()
        while not q.empty():
            node = q.get_nowait()
            if node in marked:
                continue
            marked[node] = True
            for nbr,_ in self.edges_incoming[node]:
                q.put(nbr)
        return list(marked.keys())

    def forward(self, input_value, input_nodes, output_nodes=None):
        if output_nodes is None:
            output_nodes = [node for node in self.nodes]
        completed = set()
        input_nodes = [self.nodes[node] for node in input_nodes]
        for node in input_nodes:
            for key in node.input_dict:
                assert (key in input_value), "Key {} not found in {}".format(key, input_value)
                if len(list(node.input_dict[key])) >= 2:
                    assert shapeMatcher(node.input_dict[key], input_value[key].shape), "Shape mismatch {}, expected: {} found: {}".format(key, node.input_dict[key], input_value[key].shape)

        for node in input_nodes:
            self.ready_queue.put(node.name)

        node_list = self.search_smallest_graph(output_nodes)

        while self.ready_queue.qsize() > 0:
            node = self.ready_queue.get()
            if not node in node_list:
                continue
            if node in completed:
                continue
            self.forward_node(input_value, node)
            completed.add(node)

        output_dict = {}
        for node in output_nodes:
            print(self.nodes[node].value)
            for key in self.nodes[node].value:
                output_dict[key] = self.nodes[node].value[key]

        return output_dict

    def __str__(self):
        return "{}\n".format(self.name) + str({
            "nodes" : [str(self.nodes[node]) for node in self.nodes],
            "edges_incoming" : self.edges_incoming,
            "edges_outgoing" : self.edges_outgoing,
            "transformations" : [str(self.transformations[transformation_name]) for transformation_name in self.transformations]
            })


class DynamicGraphEntity(object):
    def __init__(self, entity):
        self.entity = entity

    def update(self, input_value):
        self.final = self.entity.forward(input_value)
        return self.final

    @property
    def value(self):
        assert self.final is not None, "Update needs to be called on {} before value can be invoked".format(self.entity.name)
        return self.final

    def reset(self, session):
        self.final = None
        return self._reset(session)

    def _reset(self):
        raise NotImplementedError

    def init(self, session):
        self.final = None
        return self._init(session)

    def _init(self, session):
        raise NotImplementedError

    @property
    def input_dict(self):
        return self.entity.input_dict

    @property
    def name(self):
        return self.entity.name

class DynamicBlock(DynamicGraphEntity):
    def _reset(self, session):
        self.final = None
        self.ready = dict({name : False for name,_ in session.dynamic_graph.edges_incoming[self.entity.name]})

    def _set_ready(self, name):
        assert name in self.ready, "Can not move transformation to ready state because blocks: {} and {} are not connected".format(name, self.entity.name)
        self.ready[name] = True

    @property
    def is_ready(self):
        return len(list(set(self.ready.values()))) == 1 and True in list(self.ready.values())

    def _init(self, dynamic_graph):
        self.entity.initializer(dynamic_graph.name)

class DynamicTransformation(DynamicGraphEntity):
    def _reset(self, session_graph):
        self.cache = dict()
        self.init(session_graph)
        self.graph = session_graph.graph

    def __call__(self, block):
        if not block.name in self.cache :
            self.cache[block.name] = self.entity.forward(block.value)
        return self.cache[block.name]

    def _init(self, session):
        return self.entity.initializer(name=session.name)

class DynamicTriggerTransformation(TriggerTransformation):
    def _reset(self, session_graph):
        super(DynamicTriggerTransformation, self)._reset(session_graph)
        self.executable = False

    def update_trigger(self, *args, **kwargs):
        if self._update_trigger(*args, **kwargs):
            self.executable = True

    @property
    def is_executable(self):
        return self.executable


class Session(object):
    def __init__(self, name="", internal_graph=None):
        assert name is not None and name != "", "Session name should be something meaningful: {}".format(name)
        assert internal_graph is not None, "internal_graph can not be None"
        assert isinstance(internal_graph, BaseGraph), "internal_graph needs to be a BaseGraph, found: {}".format(internal_graph)
        self.name = name
        self.graph = internal_graph

    def init_dynamic_graph(self):
        self.dynamic_graph = DynamicGraph(self.graph)
        self.dynamic_graph.init(self)
        self.dynamic_graph.reset(self)

    def as_default(self):
        class SessionScopeHandler(object):
            def __init__(self, session):
                self.session = session

            def __enter__(self):
                self.session.init_dynamic_graph()
                self.prev_session = baseConfiguration["session"]
                baseConfiguration["session"] = self.session
                return self.session

            def __exit__(self, type, value, traceback):
                self.dynamic_graph = None
                if type is not None:
                    return
                assert hasattr(self, 'prev_session') and self.prev_session is not None, "A session was not set while scoping"
                baseConfiguration["session"] = self.prev_session
                return True
        return SessionScopeHandler(self)

    def run(self, *args, **kwargs):
        self.dynamic_graph.forward(*args, **kwargs)

    def __str__(self):
        return "{}\n".format(self.name) + str(self.graph)

baseConfiguration["session"] = Session("default-session", baseConfiguration["graph"])

if __name__ == "__main__":
    graph = BaseGraph("not_default")
    with graph.as_default() as current_graph:
        block = Block(input_dict={"img" : [288, 288, 4]}, output_dict={"img" : [144, 144, 3]})
        betadom = Block(input_dict={"betadom" : [10,10]}, output_dict={"betadom" : [None, 10]})
        transformation = Transformation(name="img-to-betadom", input_dict={"img" : [144, 144, 3]}, output_dict={"betadom" : [10, 10]})(source=block, target=betadom)
        print(current_graph)
        session = Session("not-default-session", internal_graph=current_graph)
        with session.as_default() as sess:
            print(sess)
    print(baseConfiguration["graph"])
