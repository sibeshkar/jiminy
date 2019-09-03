from jiminy.sdk.wrappers import Transformation

class Identity(Transformation):
    def __init__(self):
        super(Transformation, self).__init__(name="identity",
                input_dict=None,
                output_dict=None)

    def _forward(self, inputs):
        return inputs
