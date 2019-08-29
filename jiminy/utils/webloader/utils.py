def getStyle(webDriverObject, objectInContext):
    """
    :param webDriverObject: the driver object which can be used to execute scripts
    :param objectInContext: the object for which we want to compute the style
    TODO: cache style strings in a LRUCache
    """
    script = "return getComputedStyle(arguments[0]);"
    styleJsonRaw = webDriverObject.execute_script(script, objectInContext)
    styleJson = dict(styleJsonRaw)
    return styleJson

def checkHidden(webDriverObject, objectInContext):
    if 'hidden' == objectInContext.get_attribute("type"):
        return True
    style = getStyle(webDriverObject, objectInContext)
    if 'visibility' in style and style['visibility'] == 'hidden':
        return True
    size = objectInContext.size['height'] * objectInContext.size['width']
    if size == 0.0:
        return True
    return False

def checkInScreen(seleniumObject, screen_shape):
    if seleniumObject.location['x'] > screen_shape[0]:
        return False
    if seleniumObject.location['y'] > screen_shape[1]:
        return False
    if seleniumObject.location['x'] + seleniumObject.size['width'] > screen_shape[0]:
        return False
    if seleniumObject.location['y'] + seleniumObject.size['height'] > screen_shape[0]:
        return False
    return True


def getInputFields(webDriverObject):
    inputObjects = []
    for inputObject in webDriverObject.find_elements_by_xpath("//input"):
        if not checkHidden(webDriverObject, inputObject):
            inputObjects.append(inputObject)
    return inputObjects

def getButtonFields(webDriverObject):
    inputObjects = []
    for inputObject in webDriverObject.find_elements_by_xpath("//button"):
        if not checkHidden(webDriverObject, inputObject):
            inputObjects.append(inputObject)
    return inputObjects

def getTextFields(webDriverObject):
    inputObjects = []
    for inputObject in webDriverObject.find_elements_by_xpath("//p"):
        if not checkHidden(webDriverObject, inputObject):
            inputObjects.append(inputObject)
    for inputObject in webDriverObject.find_elements_by_xpath("//div"):
        if not checkHidden(webDriverObject, inputObject) and inputObject.text != None and inputObject.text != "":
            inputObjects.append(inputObject)
    return inputObjects

def getInstructionFields(webDriverObject):
    inputObjects = []
    for inputObject in webDriverObject.find_elements_by_id("query"):
        if not checkHidden(webDriverObject, inputObject):
            inputObjects.append(inputObject)
    assert len(inputObjects) == 1, "Expected single query object found: {}".format(len(inputObjects))
    return inputObjects[0]

