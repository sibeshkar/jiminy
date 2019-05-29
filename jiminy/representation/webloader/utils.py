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


def getInputFields(webDriverObject):
    inputObjects = []
    for inputObject in webDriverObject.find_elements_by_xpath("//input"):
        if not checkHidden(webDriverObject, inputObject):
            inputObjects.append(inputObject)
    return inputObjects

