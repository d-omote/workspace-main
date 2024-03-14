# 再帰で領域拡張法に挑戦した(未完)
# 再帰の上限に達してしまいエラー

import cv2
import numpy as np
import matplotlib.pyplot as plt

from Region import RegionGray
#import Region

class MyRegionGrow:
    def __init__(self, imgPath, targetY, targetX):
        self.inputImg = cv2.imread(imgPath)
        self.height = self.inputImg.shape[0]
        self.width = self.inputImg.shape[1]
        self.region = RegionGray(self.height, self.width)
        #判断基準
        self.criterion = 30
        
        self.regionGrowing(targetY, targetX ,targetY, targetX)
        print(self.region.getPixelNum())
        cv2.imshow("aa", cv2.cvtColor(self.region.getBinaryImg(), cv2.COLOR_GRAY2RGBA))
        cv2.waitKey(0)
        
    #2画素の類似度が高いか判断
    def isCloseBright(self, souY, souX, targetY, targetX):
        #[B, G, R]注意
        #souPix:比較元の画素
        souPix = self.inputImg[souY][souX].astype(np.int64) 
        #targetPix:比較先の画素
        targetPix = self.inputImg[targetY][targetX].astype(np.int64)
        #2画素間の距離を計算
        colorDis = (souPix[0] - targetPix[0]) ** 2 + (souPix[1] - targetPix[1]) ** 2 + (souPix[2] - targetPix[2]) ** 2
        return colorDis < self.criterion
    
    #領域拡張法
    def regionGrowing(self, souY, souX, targetY, targetX):
        #127:未探索,0:領域ではない,255:領域
        #未探索か？
        if(self.region.getBinaryImg()[targetY][targetX] == 127):
            self.region.setPix2BinaryImg(targetY, targetX, 0)
            #領域を拡張できるか？
            if self.isCloseBright(souY, souX, targetY, targetX):
                #領域拡張して、近傍の画素を探索する
                self.region.add(targetY, targetX, 255)
                self.fourNearSearch(targetY, targetX)
                #self.eightNearSearch(targetY, targetX)

    #8近傍探索
    def eightNearSearch(self, targetY, targetX):
        self.regionGrowing(targetY, targetX, max(0, targetY - 1), targetX)
        self.regionGrowing(targetY, targetX, max(0, targetY - 1), max(0, targetX - 1))
        self.regionGrowing(targetY, targetX, targetY, max(0, targetX - 1))
        self.regionGrowing(targetY, targetX, min(targetY + 1, self.height - 1), min(0, targetX - 1))
        self.regionGrowing(targetY, targetX, min(targetY + 1, self.height - 1), targetX)
        self.regionGrowing(targetY, targetX, min(targetY + 1, self.height - 1), min(targetX + 1 , self.width - 1))
        self.regionGrowing(targetY, targetX, targetY, min(targetX + 1, self.width - 1))
        self.regionGrowing(targetY, targetX, max(0, targetY - 1), min(targetX + 1, self.width))
        
    #4近傍探索
    def fourNearSearch(self, targetY, targetX):
        self.regionGrowing(targetY, targetX, max(0, targetY - 1), targetX)
        self.regionGrowing(targetY, targetX, targetY, max(0, targetX - 1))
        self.regionGrowing(targetY, targetX, min(targetY + 1, self.height - 1), targetX)
        self.regionGrowing(targetY, targetX, targetY, min(targetX + 1, self.width - 1))
            
            
    