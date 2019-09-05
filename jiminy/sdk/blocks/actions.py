from jiminy.sdk.wrappers import Block
from bs4 import BeautifulSoup
import urllib.request as urllib2

class SelectedTextBlock(Block):
    def __init__(self, name=""):
        super(SelectedTextBlock, self).__init__(input_dict={"selected-text" : tuple(), "url" : tuple()},
            output_dict={"selected-text" : tuple(), "url" : tuple(), "title" : ()},
            name = name + "SelectedTextBlock")

    def _forward(self, inputs):
        selected_text = inputs["selected-text"]
        url = inputs["url"]
        if not("https" in url or "http" in url):
            url = "http://" + url
        soup = BeautifulSoup(urllib2.urlopen(url), features="html.parser")
        inputs["title"] = soup.title.string
        return inputs

if __name__ == "__main__":
    selected = SelectedTextBlock()
    output = selected.forward({
        "selected-text" : "hi",
        "url" : "https://www.google.com/"
        })
    print(output)
