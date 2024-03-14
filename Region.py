##筆跡(領域を管理する)

import numpy as np
import cv2

#BGR画像
class RegionColor:
    def __init__(self, height, width):
        #領域内の画素の座標(y,x)のリスト
        self.pixPosList = []
        #画素値が0の部分は背景
        #self.bgrImg = np.full((height, width, 3), 0, np.uint8)
        
    def getPixelNum(self):
        return len(self.pixPosList)
         
    def getPixPosList(self):
        return self.pixPosList
    
    #def getBinaryImg(self):
    #    return self.bgrImg
    
    def addPixelPosList(self, y, x):
        self.pixPosList.append([y, x])
    
    #def setPix2BinaryImg(self, y, x, value):
    #    self.bgrImg[y][x] = value
    
    def add(self, y, x, value):
        self.addPixelPosList(y, x)
        #self.bgrImg[y][x] = value
        
    def show(self):
        #cv2.imshow('Result', self.bgrImg)
        cv2.waitKey(0)
        
    def save(self):
        #cv2.imwrite("./result/region.jpg", self.bgrImg)
        print("保存しました")

#グレースケール画像
class RegionGray:
    def __init__(self, height, width):
        #領域内の画素の座標(y,x)のリスト
        self.pixelPosList = []
        #127は未探索の意味(0は背景、255は領域)
        self.binaryImg = np.full((height, width), 127, np.uint8)
    #領域を構成する画素の数を返す
    def getPixelNum(self):
        return len(self.pixelPosList)
         
    def getPisPosList(self):
        return self.pixelPosList
    #領域を表す画像
    def getBinaryImg(self):
        return self.binaryImg
    
    def addPixelPosList(self, y, x):
        self.pixelPosList.append([y, x])
    
    def setPix2BinaryImg(self, y, x, value):
        self.binaryImg[y][x] = value
    
    def add(self, y, x, value):
        self.addPixelPosList(y, x)
        self.binaryImg[y][x] = value