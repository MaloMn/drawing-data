import cv2 as cv
import numpy as np
from skimage.morphology import skeletonize, medial_axis, thin
from pathfinding import PathFinder
from itertools import combinations


def redraw_shape(img):
    output = np.zeros(img.shape, np.uint8)
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda a: cv.contourArea(a), reverse=True)
    contours = [a for a in contours if cv.contourArea(a) > 50]
    contours = [cv.approxPolyDP(a, 2, True) for a in contours]
    # contours = [cv.convexHull(a) for a in contours]

    cv.fillPoly(output, [contours[0]], 255)

    return output


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


def _loose_ends(points, path_finder):
    output = []
    for pt in points:
        # print(pt, path_finder.get_neighbours(pt))
        if len(path_finder.get_neighbours(pt)) == 1:
            output.append(pt)
    return output


def longest_chain(points, path_finder):
    ends = _loose_ends(points, path_finder)
    print(ends)
    return max([path_finder.find_path(a, b) for a, b in combinations(ends, 2)], key=lambda a: len(a))


if __name__ == "__main__":
    filename = 'line.png'
    name = filename.split('.')[0]
    source_folder = 'photos/'
    dest = 'debug/'

    original = cv.imread(source_folder + filename, 0)

    # Simple thresholding
    _, thresh = cv.threshold(original, 75, 255, cv.THRESH_BINARY)
    thresh = cv.bitwise_not(thresh)
    cv.imwrite(dest + name + '_a_thresh.jpg', thresh)

    thresh = redraw_shape(thresh)
    # cv.imshow('redrawn', thresh)
    cv.imwrite(dest + name + '_b_redrawn.jpg', thresh)

    line = extract_lines(thresh, method='classic')
    # cv.imshow("Lee", line)
    cv.imwrite(dest + name + "_c_skeleton_lee.jpg", line)

    line = np.where(line == 255)
    pts = []
    for i, j in zip(*line):
        pts.append((int(i), int(j)))
    line = pts

    finder = PathFinder(set(line), original.shape)
    chain = longest_chain(line, finder)
    rows, cols = zip(*chain)

    line_draw = np.zeros(original.shape, np.uint8)
    line_draw[rows, cols] = 255
    cv.imwrite(dest + name + '_d_chain.jpg', line_draw)

    cv.waitKey(0)
    cv.destroyAllWindows()

