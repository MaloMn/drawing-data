import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('photos/weird-bug.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Simple thresholding
_, thresh1 = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)

# Adaptive thresholding
thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Filtering out blue
ink_color = np.array([159, 180, 41])
thresh = 115
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = ink_color - (thresh,) * 3
upper = ink_color + (thresh,) * 3
print(lower, upper)
thresh3 = cv2.inRange(hsv, lower, upper)
# print(hsv)

# plt.imshow(thresh3)

# Displaying
titles = ['Original Image', 'BINARY', 'ADAPTIVE', 'COLOR FILTER']
images = [img, thresh1, thresh2, thresh3]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

plt.show()
