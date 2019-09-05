from jiminy.sdk.wrappers import Block

class MemoryNode(Block):
    def __init__(self, memory_type="queue"):
        self.memory_type = memory_type
        if self.memory_type == "queue":
            self.memory_object = queue.Queue(maxsize=10)
        elif self.memory_object == "priority_queue":
            self.memory_object = queue.PriorityQueue(maxsize=10)

    def _forward(self, inputs):
        assert len(inputs) == 1, "input to MemoryNode needs to be a single variable that can be logged"
        for key in inputs:
            if self.memory_object.full():
                _ = self.memory_object.get()
            self.memory_object.put(inputs[0])
