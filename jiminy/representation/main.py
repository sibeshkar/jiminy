from selenium import webdriver
import utils
import argparse
import os
from webloader import WebLoader
from jiminy.representation.structure import JiminyBaseObject,betaDOMFromSeleniumWebDriver
from jiminy.vectorized.core import Env


parser = argparse.ArgumentParser("Jiminy web loader module")
parser.add_argument('--urlFile',dest='urlFile',
        action='store', default='url_list.txt',
        help='File which consists list of URLs to be inferred and stored as representation objects')
args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.urlFile):
        raise ValueError("File {} not found while trying to load URLs to create representation objects from".format(args.urlFile))
    urlList = []
    with open(args.urlFile, mode="r") as f:
        urlList += f.readlines()
    urlList = [i.replace('\n', '') for i in urlList]
    webLoader = WebLoader.WebLoader("Firefox")
    for url in urlList:
        webLoader.loadPage(url)
        betadom = betaDOMFromSeleniumWebDriver(webLoader.driver)
        objectList = webLoader.getRawObjectList()
        objectList = [JiminyBaseObject(betaDom=betadom, seleniumObject=seleniumObj) for seleniumObj in objectList]
        for jimObj in objectList:
            print(jimObj.toString())
    webLoader.driver.close()
