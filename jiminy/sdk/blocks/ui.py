from jiminy.sdk.wrappers import Block
import numpy as np

class PossibleActionList(Block):
    def __init__(self, actionList, ui_handler, name=""):
        super(PossibleActionList, self).__init__(name=name+"possible-action-list",
                input_dict={
                    "title" : (0,),
                     "url" : (0,),
                     "selected_text" : (0,)
                    },
                output_dict={
                    "action" : (0,)
                    })
        assert isinstance(actionList, list), "Expected actionlist to be of type list, found: {}".format(type(actionList))
        for elem in actionList:
            assert isinstance(elem, str), "Expected action to be possible action encoded as string, found: {}".format(elem)
        self.actionList = actionList
        self.ui_handler = ui_handler

    def _forward(self, inputs):
        actionList = self._filter_possible_actions(inputs)
        action = self.ui_handler(actionList)
        # insert code to communicate with UI
        # action = run API query
        return {
                "action" : action
                }

    def _filter_possible_actions(self, inputs):
        if len(self.actionList) > 4:
            selected_text = inputs["selected_text"]
            if len(selected_text) > 100:
                # for small text we only have non-summary based options
                return self.actionList[:4]
            else:
                # for larger texts we have summarization options
                return self.actionList[4:]
        return self.actionList

if __name__ == "__main__":
    def ui_handler_fn(actionList):
        return np.random.choice(actionList)
    pal = PossibleActionList(["Add to Flow", "Add to Note", "Add to Drive", "Add to Notion",
        "Summarize to Flow", "Summarize to Note", "Summarize to Drive", "Summarize to Notion"],
        ui_handler=ui_handler_fn)
    pal.forward(inputs={
        "title" : "Google",
        "url" : "https://google.com/",
        "selected_text" : "hi"
        })
