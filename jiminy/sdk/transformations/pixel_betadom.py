from jiminy.sdk.wrappers import Transformation
import numpy as np
import cv2
from PIL import Image
import pytesseract as pt
import re

class PixelToBetadomTransformation(Transformation):
    def __init__(self, font_family='courier', img_shape=[None, None, 3]):
        super(PixelToBetadomTransformation, self).__init__(name="pixel-to-betadom-{}".format(font_family),
                input_dict={"img" : img_shape},
                output_dict={"betadom" : [None, 10]})
        self.font_family = font_family
        self.text_regex = '[a-zA-Z0-9 ]*'

class PixelToSelectedText(Transformation):
    def __init__(self, image_shape=[None, None, 3]):
        super(PixelToSelectedText, self).__init__(name="pixel-to-selected-text",
                input_dict={"img" : image_shape},
                output_dict={"selected-text" : (1,)})

    def _forward(self, inputs):
        img = inputs["img"]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([50, 50, 20])
        upper_red = np.array([255, 255, 255])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask_covering = np.zeros(mask.shape, np.uint8)
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size > 500:
                cv2.drawContours(mask_covering, [cnt], -1, (255,255,255), -1)

        mask = mask * mask_covering

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask_covering = np.zeros(mask.shape, np.uint8)
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size < 1000:
                continue
            cv2.drawContours(mask_covering, [cnt], -1, (255,255,255), -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        mask_covering = cv2.dilate(mask_covering, kernel, iterations=8)

        res = cv2.bitwise_and(img, img, mask=mask_covering)
        grayscale = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pt.image_to_string(res)
        return {
                "selected-text" : text
                }

class PixelToElements(Transformation):
    def __init__(self, image_shape=[None, None, 3]):
        super(PixelToElements, self).__init__(name="pixel-to-selected-text",
                input_dict={"img" : image_shape},
                output_dict={"parts" : (None,) + tuple(image_shape)})

    def _forward(self, inputs):
        img = inputs["img"]
        if "inverse" in inputs and inputs["inverse"]:
            inverse = cv2.THRESH_BINARY_INV
        else :
            inverse = cv2.THRESH_BINARY
        seperated_parts = self._forward_implementation(img, inverse)
        return {
                "parts" : seperated_parts
                }

    def _forward_implementation(self, img, inverse):
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(grayscale, 0, 255, inverse + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,2))
        dilated = cv2.dilate(threshold, kernel, iterations=15)
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        seperated_parts = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            if (h < 100 and w < 100) or area < 1000:
                continue
            if h == img.shape[0] and w == img.shape[1]:
                continue

            seperated_parts.append((img[y:y+h,x:x+w], x, y, w, h))

        if inverse == cv2.THRESH_BINARY :
            reverse_invert = cv2.THRESH_BINARY_INV
        else :
            reverse_invert = cv2.THRESH_BINARY

        seperated_part_list = []
        while len(seperated_parts):
            seperated_part_running = []
            for part in seperated_parts:
                parts = self._forward_implementation(part[0], reverse_invert)
                if parts == []:
                    seperated_part_list.append(part)
                    continue
                else:
                    seperated_part_running += parts
            seperated_parts = seperated_part_running

        return seperated_part_list

class PixelToURL(Transformation):
    def __init__(self, image_shape=[None, None, 3], url_regex=''):
        super(PixelToURL, self).__init__(name="pixel-to-URL-transformation",
                input_dict={"img" : image_shape},
                output_dict={"url" : tuple([])})
        if len(url_regex) > 4:
            self.url_regex = url_regex
        else:
            self.url_regex = r"\w*(?:(?:\.\w*)+)(?:(?:/(?:[\w\.]*))*)(?:.\w*)"


    def _forward(self, inputs):
        img = inputs["img"]
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(grayscale[80:160], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        topbar_text = pt.image_to_string(threshold)
        url_list = re.findall(self.url_regex, topbar_text)
        if len(url_list) < 1:
            raise ValueError("No URLs found in browser search bar")
        return {
                "url" : url_list[0]
                }


def process_contour(index, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img-gray-{}.png'.format(index), gray)
    edges = cv2.Canny(gray, 60, 200)
    _, res = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('img-edges-{}.png'.format(index), res + edges)
    contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask_new = np.ones(img.shape, np.uint8) * 255
    for cnt in contours:
        size = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        if size < 10000 and w*2.5 > h:
            cv2.drawContours(mask_new, [cnt], -1, (0,0,0), -1)
    cv2.imwrite('img-mask-{}.png'.format(index), mask_new)

    kernel = np.ones((50,50),np.uint8)
    opening = cv2.morphologyEx(mask_new, cv2.MORPH_OPEN, kernel)
    gray_op = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    _, threshold_op = cv2.threshold(gray_op, 150, 255, cv2.THRESH_BINARY_INV)
    contours_op, hierarchy_op = cv2.findContours(threshold_op, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cnt = max(contours_op, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.imwrite('text-area-{}.png'.format(index), img[y:y+h,x:x+w])

    cv2.imwrite('img-{}.png'.format(index), res)
    read_text = pt.image_to_string('img-{}.png'.format(index))
    read_text = ' '.join(read_text.split())
    read_text = ''.join(e for e in read_text if e.isalnum() or e == ' ' or e in ".-_?/@")
    read_text = read_text.split(' x ')
    print(read_text)

if __name__ == "__main__":
    inputs = {
            "img" : cv2.imread("/Users/prannayk/Desktop/keep-title.png")
            }
    pix2text = PixelToSelectedText()
    print(pix2text.forward(inputs))
    pix2parts = PixelToElements()
    pix2url = PixelToURL()
    print(pix2url.forward(inputs))
    for idx, part in enumerate(pix2parts.forward(inputs)["parts"]):
        cv2.imwrite("part-{}.png".format(idx), part[0])
        text = pt.image_to_string(part[0])
        print(text)
        if "Take a note" in text or "Title" in text:
            print(part[1:])
