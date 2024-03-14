from PIL import Image, ImageDraw

import random
import math

from torch import true_divide

def main(inputImage):
    radius = 3
    width, height = inputImage.size
    sampleNum = math.floor(width * height /(2 * radius * radius))
    print(sampleNum)
    resultImage = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(resultImage)
    circleList = []
    missTake = 0
    counter = 0
    n = 1
    progress = 0

    while sampleNum > counter and missTake < sampleNum:
        x = random.randint(0, width -1)
        y = random.randint(0, height -1)
        foundOverlap = False

        for circle in circleList:
            if (x - circle.getX()) * (x - circle.getX()) + (y - circle.getY()) * (y - circle.getY()) <= n * (radius + circle.getRadius()) * (radius + circle.getRadius()):
                foundOverlap = True
                missTake += 1
                break
        
        if foundOverlap == False:
            circleList.append(Circle(x, y, radius))
            missTake = 0
            counter += 1
            buffer = progress
            progress = math.floor(counter / sampleNum * 100)
            if(progress > buffer and progress % 10 == 0):
                print(progress, "%・・・")
    
    for circle in circleList:
        color = inputImage.getpixel((circle.getX(), circle.getY()))
        draw.ellipse((circle.getX() - circle.getRadius(), circle.getY() - circle.getRadius(), circle.getX() + circle.getRadius(), circle.getY() + circle.getRadius()), fill=(color))

    return resultImage

class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getRadius(self):
        return self.radius
