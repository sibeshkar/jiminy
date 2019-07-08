from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from jiminy.utils.webloader import utils

class WebLoader:
    def __init__(self, browser='Firefox'):
        if browser == 'Firefox':
            options = Options()
            options.headless = True
            self.driver = webdriver.Firefox(options=options)
        else:
            raise ValueError("Unknown broswer configuration: {}".format(browser))
        self.active_object = None

    def loadPage(self, url):
        self.driver.get(url)

    def getRawObjectList(self, screen_shape=None):
        """
        Warning: do not call this without calling load page first
        :returns: a list of raw selenium objects which are relevant to the information being inferred by the engine
        """
        objectList = []
        objectList += utils.getInputFields(self.driver)
        objectList += utils.getButtonFields(self.driver)
        objectList += utils.getTextFields(self.driver)

        if screen_shape:
            objectList = [obj for obj in objectList if utils.checkInScreen(obj, screen_shape)]
        return objectList

    def getInstructionFields(self):
        return utils.getInstructionFields(self.driver)
