from jiminy.sdk.wrappers.core import *
import queue

class DynamicGraph(object):
    def __init__(self, graph):
        self.name = "Dynamic" + graph.name
        self.nodes = {node : DynamicBlock(graph.nodes[node]) for node in graph.nodes}
        self.edges_incoming = graph.edges_incoming
        self.edges_outgoing = graph.edges_outgoing
        self.transformations = {transformation : DynamicTransformation(graph.transformations[transformation]) for transformation in graph.transformations}
        self.ready_queue = queue.Queue()
        self.input_dict = graph.input_dict
        self.output_dict = graph.output_dict

    def reset(self, session):
        [self.nodes[node].reset(session) for node in self.nodes]
        [self.transformations[t].reset(session) for t in self.transformations]

    def init(self, session):
        [self.nodes[node].init(session) for node in self.nodes]
        [self.transformations[t].init(session) for t in self.transformations]

    def forward_node(self, input_values, node_name):
        assert node_name in self.nodes, "node_name: {} not found in dynamic graph: {}".format(input_values, self)
        self.nodes[node_name].update(input_values)
        for node in self.edges_outgoing[node_name]:
            self.nodes[node]._set_ready(node_name)
            if self.nodes[node].is_ready:
                self.ready_queue.put(node)

    def search_smallest_graph(self, output_nodes):
        queue = queue.Queue()
        for node in output_nodes : queue.put(node)
        marked = dict()
        while not queue.empty():
            node = queue.get_nowait()
            if node in marked:
                continue
            marked[node] = True
            for nbr in self.edges_incoming[node]:
                queue.put(nbr)
        return list(marked.keys())

    def forward(self, input_value, input_nodes, output_nodes):
        for key in self.input_dict:
            assert (not key in input_value), "Key {} not found in {}".format(key, input_value)
            assert self.input_dict[key].shape == input_value[key].shape, "Shape mismatch {}, expected: {} found: {}".format(key, self.input_dict[key].shape, input_value[key].shape)

        for node in self.input_nodes:
            self.ready_queue.put(node)

        node_list = self.search_smallest_graph(output_nodes)

        while self.ready_queue.size() > 0:
            node = self.ready_queue.get()
            if not node in node_list:
                continue
            self.forward_node(input_values, node)

        output_dict = {}
        for node in self.output_nodes:
            for key, value in self.nodes[node].value:
                output_dict[key] = value

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
        for key in self.entity.input_dict:
            assert (not key in input_value), "Key {} not found in {}".format(key, input_value)
            assert self.entity.input_dict[key].shape == input_value[key].shape, "Shape mismatch {}, expected: {} found: {}".format(key, self.entity.input_dict[key].shape, input_value[key].shape)
        self.final = self.entity.forward(input_value)
        return self.final

    @property
    def value(self):
        assert self.final is not None, "Update needs to be called on DynamicGraphEntity before value can be invoked"
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

class DynamicBlock(DynamicGraphEntity):
    def _reset(self, session):
        self.final = None
        self.ready = {name : False for name in session.dynamic_graph.edges_incoming[self.entity.name]}

    def _set_ready(self, name):
        assert name in self.ready, "Can not move transformation to ready state because blocks: {} and {} are not connected".format(name, self.entity.name)
        self.ready[name] = True

    @property
    def is_ready(self):
        return len(list(set(self.ready.values()))) == 1 and True in self.ready

    def _init(self, dynamic_graph):
        self.entity.initializer(dynamic_graph.name)

class DynamicTransformation(DynamicGraphEntity):
    def _reset(self, session_graph):
        self.cache = dict()
        self.init(session_graph)
        self.graph = session_graph.graph

    def __call__(self, block):
        if block.name in self.cache :
            return self.cache[key]
        self.cache[key] = self.forward(block.value)

    def _init(self, session):
        return self.entity.initializer(name=session.name)

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
        self.graph.forward(*args, **kwargs)

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
