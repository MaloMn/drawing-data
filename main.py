import cv2 as cv
import numpy as np
from pathfinding import PathFinder
from lines import redraw_shape, extract_lines, longest_chain, save_json

source_folder = 'photos/'
dest = 'debug/'


def analyse_figure(filename, threshold=75, method='classic', debug=False):
    name = filename.split('.')[0]
    original = cv.imread(source_folder + filename, 0)

    # Simple thresholding
    _, thresh = cv.threshold(original, threshold, 255, cv.THRESH_BINARY)
    thresh = cv.bitwise_not(thresh)
    if debug:
        cv.imwrite(dest + name + '_a_thresh.jpg', thresh)

    thresh = redraw_shape(thresh)
    if debug:
        cv.imwrite(dest + name + '_b_redrawn.jpg', thresh)

    line = extract_lines(thresh, method=method)
    if debug:
        cv.imwrite(dest + name + "_c_skeleton.jpg", line)

    line = np.where(line == 255)
    pts = []
    for i, j in zip(*line):
        pts.append((int(i), int(j)))
    line = pts

    finder = PathFinder(set(line), original.shape)
    chain = longest_chain(line, finder)
    save_json(chain, name)

    rows, cols = zip(*chain)

    line_draw = np.zeros(original.shape, np.uint8)
    line_draw[rows, cols] = 255
    if debug:
        cv.imwrite(dest + name + '_d_chain.jpg', line_draw)


if __name__ == "__main__":
    import os
    a = len(os.listdir(source_folder))
    for i, file in enumerate(os.listdir(source_folder)):
        print(f"({i + 1}/{a}) {file}")
        analyse_figure(file, debug=True)
        i += 1
