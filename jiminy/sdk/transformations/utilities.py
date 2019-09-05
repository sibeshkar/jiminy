from jiminy.sdk.wrappers import Transformation

class Filter(Transformation):
    def __init__(self, filter_regex="*"):
        self.filter = re.compile(filter_regex)
        super(Transformation, self).__init__(name="identity",
                input_dict=None,
                output_dict=None)

    def _forward(self, inputs):
        output = dict()
        for key in inputs:
            assert isinstance(key, str), "key to input dictionary needs to be a string, found: {}".format(key)
            match = self.filter.match(key)
            if len(match) == 1 and match[0] == key:
                output[key] = inputs[key]
        return output


class Identity(Transformation):
    def __init__(self):
        super(Transformation, self).__init__(name="identity",
                input_dict=None,
                output_dict=None)

    def _forward(self, inputs):
        return inputs
