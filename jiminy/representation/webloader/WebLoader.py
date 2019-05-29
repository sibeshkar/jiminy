from selenium import webdriver
from webloader import utils

class WebLoader:
    def __init__(self, browser='Firefox'):
        if browser == 'Firefox':
            self.driver = webdriver.Firefox()
        else:
            raise ValueError("Unknown broswer configuration: {}".format(browser))

    def loadPage(self, url):
        self.driver.get(url)

    def getRawObjectList(self):
        """
        Warning: do not call this without calling load page first
        :returns: a list of raw selenium objects which are relevant to the information being inferred by the engine
        """
        objectList = []
        objectList += utils.getInputFields(self.driver)
        return objectList

