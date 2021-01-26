import cv2 as cv
import numpy as np
from typing import List
from matplotlib import pyplot as plt

original = cv.imread('photos/line.png', 0)

# Simple thresholding
_, thresh = cv.threshold(original, 75, 255, cv.THRESH_BINARY)
thresh = cv.bitwise_not(thresh)

# cv.imshow("img", thresh)


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
# cv.imshow('redrawn', thresh)

# Extracting the lines from the
line = cv.ximgproc.thinning(thresh)
# cv.imshow("Skeleton", line)


def get_longest_line(lines):
    indexes = np.where(lines == 255)
    pts = []
    for i, j in zip(*indexes):
        pts.append([i, j])

    pts.sort(key=lambda a: sum(a))

    start = pts[0]
    pts.remove(start)

    return _longest_line(start, pts, 1)


def _longest_line(point, points, length):
    closest_points = filter(lambda a: abs(a[0] - point[0]) + abs(a[1] - point[1]), points)


print(get_longest_line(line))


cv.waitKey(0)
cv.destroyAllWindows()

