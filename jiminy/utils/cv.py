"""
Implements classic computer vision algorithms to
process webpages and infer objects from it
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def detect_rectangle_from_contour(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05*peri, True)
    area = cv2.contourArea(contour)
    if area < 500:
        return False
    if len(approx) <= 30 and len(approx) >= 4:
        return True
    return False

def load_objects(fname="{}/jiminy/agents/20190625-233334.png".format(os.getenv("JIMINY_ROOT"))):
    img_raw = cv2.imread(fname)
    edges_raw = cv2.Canny(img_raw, 30, 220)
    contour_list, hier = cv2.findContours(edges_raw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    edges_raw = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hier = cv2.findContours(edges_raw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours += contour_list
    hsv = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    th, threshed = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)
    contour_list,_ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours += contour_list
    # contours = list(filter(lambda c : detect_rectangle_from_contour(c), contours))
    for contour in contours:
        img = np.copy(img_raw)
        plt.imshow(img)
        plt.show()
        if not detect_rectangle_from_contour(contour):
            continue
        cv2.drawContours(img, [contour], -1, (0,255,0), 3)
        plt.imshow(img)
        plt.show()
    # plt.subplot(122), plt.imshow(img_raw)


if __name__ == "__main__":
    load_objects()
    # plt.show()
