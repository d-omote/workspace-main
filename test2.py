import cv2
import numpy as np

img = cv2.imread('./content/test/bustshot.jpg')

cv2.fillConvexPoly(img, np.array([(210, 200), (220, 300), (300, 340), (340, 220)]), (255, 0, 0))
cv2.imshow('output.jpg', img)
cv2.waitKey()