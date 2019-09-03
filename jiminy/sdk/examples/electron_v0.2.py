from jiminy.sdk.wrappers import BaseGraph
from jiminy.sdk.transformations import PixelToSelectedText, PixelToURL, Identity
from jiminy.sdk.blocks import ImageDataBlock, PossibleActionList, SelectedTextBlock
import numpy as np

def dummy_ui_handler_fn(actionList):
    return np.random.choice(actionList)

if __name__ == "__main__":
    graph = BaseGraph("text-selection-to-note")
    with graph.as_default():
        dataBlock = ImageDataBlock(name="main-data-block")
        pix2selectedText = PixelToSelectedText()
        pix2url = PixelToURL()
        selectedTextBlock = SelectedTextBlock("selected-text")
        chosenActionBlock = PossibleActionList(name="action-list", actionList=["Add to Flow", "Add to Note", "Add to Drive", "Add to Notion",
            "Summarize to Flow", "Summarize to Note", "Summarize to Drive", "Summarize to Notion"],
            ui_handler=dummy_ui_handler_fn)
        pix2selectedText(dataBlock, selectedTextBlock)
        pix2url(dataBlock, selectedTextBlock)
        Identity()(selectedTextBlock, chosenActionBlock)
