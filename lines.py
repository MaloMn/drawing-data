import cv2 as cv
import numpy as np
from typing import List
from matplotlib import pyplot as plt

original = cv.imread('photos/line.png', 0)

# Simple thresholding
_, thresh = cv.threshold(original, 75, 255, cv.THRESH_BINARY)
thresh = cv.bitwise_not(thresh)

cv.imshow("img", thresh)


def redraw_shape(img):
    output = np.zeros(img.shape, np.uint8)
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda a: cv.contourArea(a), reverse=True)
    contours = [a for a in contours if cv.contourArea(a) > 50]
    contours = [cv.approxPolyDP(a, 2, True) for a in contours]
    # contours = [cv.convexHull(a) for a in contours]

    cv.fillPoly(output, [contours[0]], 255)

    return output


thresh = redraw_shape(thresh)
cv.imshow('redrawn', thresh)


def find_skeleton(img):
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skeleton = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    # Repeat steps 2-4
    while True:
        # Step 2: Open the image
        open = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        # Step 3: Subtract open from the original image
        temp = cv.subtract(img, open)
        # Step 4: Erode the original image and refine the skeleton
        eroded = cv.erode(img, element)
        skeleton = cv.bitwise_or(skeleton, temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv.countNonZero(img) == 0:
            break

    return skeleton


# line = find_skeleton(thresh)
line = cv.ximgproc.thinning(thresh)
cv.imshow("Skeleton", line)


def get_longest_line(lines):
    

cv.waitKey(0)
cv.destroyAllWindows()

