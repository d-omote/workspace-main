"""
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('./otherImage/style.jpg')
img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
colors = ("r", "g", "b")
for i, channel in enumerate(colors):
    histgram = cv2.calcHist([img_1], [i], None, [256], [0, 256])
    plt.plot(histgram, color = channel)
    plt.xlim([0, 256])
plt.show()
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def color_hist(filename):
    img = np.asarray(Image.open('./otherImage/Mone.jpg').convert("L")).reshape(-1,1)
    plt.hist(img, bins=128)
    plt.show()

color_hist("./test.jpg")
