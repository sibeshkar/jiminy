from selenium import webdriver
from jiminy.utils.webloader import utils

class WebLoader:
    def __init__(self, browser='Firefox'):
        if browser == 'Firefox':
            self.driver = webdriver.Firefox()
        else:
            raise ValueError("Unknown broswer configuration: {}".format(browser))
        self.active_object = None

    def loadPage(self, url):
        self.driver.get(url)

    def getRawObjectList(self):
        """
        Warning: do not call this without calling load page first
        :returns: a list of raw selenium objects which are relevant to the information being inferred by the engine
        """
        objectList = []
        objectList += utils.getInputFields(self.driver)
        objectList += utils.getButtonFields(self.driver)
        objectList += utils.getTextFields(self.driver)
        return objectList

    def getInstructionFields(self):
        return utils.getInstructionFields(self.driver)
