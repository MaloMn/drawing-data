import cv2 as cv
import numpy as np
from typing import List
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize, medial_axis, thin
from skimage import data


original = cv.imread('photos/line.png', 0)

# Simple thresholding
_, thresh = cv.threshold(original, 75, 255, cv.THRESH_BINARY)
thresh = cv.bitwise_not(thresh)


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


def extract_lines(binary, method='classic'):
    binary = np.where(binary >= 1, 1, 0)
    if method == 'classic':
        output = skeletonize(binary)
    elif method == 'lee':
        output = skeletonize(binary, method='lee')
    elif method == 'thin':
        output = thin(binary)
    else:
        output = medial_axis(binary, return_distance=False)

    output = np.where(output, 255, 0)
    output = output.astype(np.uint8)
    return output


line1 = extract_lines(thresh)
cv.imshow("Classic", line1)

line2 = extract_lines(thresh, method='lee')
cv.imshow("Lee", line2)

line3 = extract_lines(thresh, method='medial')
cv.imshow("Medial", line3)

line4 = extract_lines(thresh, method='thin')
cv.imshow("thin", line4)

cv.imshow("product", line1*line2*line3*line4)


# indexes = np.where(line == 255)
# pts = []
# for i, j in zip(*indexes):
#     pts.append([i, j])
# line = pts

#
# def get_longest_line(lines):
#     lines.sort(key=lambda a: abs(sum(a)))
#     start = lines[0]
#     lines.remove(start)
#
#     return _longest_line(start, lines, [start], 0)[0]
#
#
# def _is_neighbour(base, pt):
#     offsets = [[1, 0], [1, -1], [1, 1], [0, -1], [0, 1], [-1, 0], [-1, 1], [-1, -1]]
#     if pt in [[base[0] + a[0], base[1] + a[1]] for a in offsets]:
#         return True
#     return False
#
#
# def _longest_line(point, points, path, length):
#     closest_points = list(filter(lambda a: _is_neighbour(point, a), points))
#     # print(closest_points, type(closest_points))
#     closest_points = [p for p in closest_points if p not in path]
#
#     if len(closest_points) == 0:
#         return path, length + 1
#
#     return max([_longest_line(pt, points.copy(), path + [pt], length + 1) for pt in closest_points], key=lambda a: a[1])


# line = [[0, 0], [0, -1], [0, -2], [1, 0], [2, 0], [0, 6], [3, 0], [2, 1], [1, 1]]

# print(get_longest_line(line))


cv.waitKey(0)
cv.destroyAllWindows()

