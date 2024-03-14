# coding: utf-8

import numpy as np
import cv2
import random

from RegionGrow import RegionColorGrow as Rc 

#画像1枚全体を領域分割するクラス
class RegionColorList:
    def __init__(self, imgPath):
        self.imgPath = imgPath
        self.inputImg = cv2.imread(imgPath)
        #self.regionList = []
        self.regionPosList = []
    
    #リストに領域追加
    def add(self, region):
        #self.regionList.append(region)
        self.regionPosList.append(region.getPixPosList())
    
    #画像全体を領域分割
    def setRegionList(self):
        height = self.inputImg.shape[0]
        width = self.inputImg.shape[1]
        #訪問済みリストをBoolean型で管理
        visitedPixPosList = np.zeros((height, width), dtype=bool) 
        tarY = 0
        tarX = 0
        #領域拡張の条件用の閾値
        #threshold = 400
        
        #画像内での最初の領域分割
        #初回なので訪問済みリストを渡さない
        region = Rc.fourNeighbor_first(self.imgPath, tarY, tarX)
        self.add(region)
        for pixPos in region.getPixPosList():
            visitedPixPosList[pixPos[0], pixPos[1]] = True  
        
        #その後はループ
        while True:
            #未訪問であれば(領域分割されていない画素であれば)領域分割実行
            if not visitedPixPosList[tarY, tarX]:
                region = Rc.fourNeighbor(self.imgPath, tarY, tarX, visitedPixPosList)
                self.add(region)
                #訪問済みリストの更新
                for pixPos in region.getPixPosList():
                    visitedPixPosList[pixPos[0], pixPos[1]] = True 
            #訪問済みor領域分割後は、未訪問の画素を探索
            if tarX < width - 1:
                tarX += 1
            elif tarY < height - 1:
                tarX = 0
                tarY += 1
            else:
                return
    
    #BGR画像で領域分割を表示
    def showRegionImg(self):
        height = self.inputImg.shape[0]
        width = self.inputImg.shape[1]
        regionImg = np.zeros((height, width, 3), np.uint8)
        regionNum = len(self.regionPosList)
        print(regionNum)
        for posList in self.regionPosList:
            B = random.randrange(256)
            G = random.randrange(256)
            R = random.randrange(256)
            for pixPos in posList:
                 regionImg[pixPos[0]][pixPos[1]] = [B, G, R]
        cv2.imshow(str(regionNum), regionImg)
        cv2.imshow("input", self.inputImg)
        cv2.waitKey(0)
        cv2.imwrite("./result/regionGrow/regionlist.jpg", regionImg)
    

    #画像全体を領域分割(訪問済みを座標リストで管理してたやつ)
    """
    
    def beforesetRegionList(self):
        height = self.inputImg.shape[0]
        width = self.inputImg.shape[1]
        visitedPixposList = []
        
        #最初は無条件に領域拡張してから、while文がいいかも
        tarY = 0
        tarX = 0
        threshold = 1
        
        region = Rc.fourNeighbor(self.imgPath, tarY, tarX, threshold)
        self.add(region)
        visitedPixposList.extend(region.getPixPosList())
        
        while True:
            if not [tarY, tarX] in visitedPixposList:
                region = Rc.fourNeighbor2(self.imgPath, tarY, tarX, threshold, visitedPixposList)
                self.add(region)
                visitedPixposList.extend(region.getPixPosList())  
                #print([tarY, tarX])
            if tarX < width - 1:
                tarX += 1
            elif tarY < height - 1:
                tarX = 0
                tarY += 1
            else:
                return
    """
   
    #グレースケール画像で領域を表示
    """
    def before_showRegionImg(self):
        height = self.inputImg.shape[0]
        width = self.inputImg.shape[1]
        testImg = np.full((height, width), 0, np.uint8)
        regionNum = len(self.regionPosList)
        print(regionNum)
        a = 255 / regionNum
        b = 0
        for posList in self.regionPosList:
            #value = int(a * b) 
            value = random.randrange(256)
            for pixPos in posList:
                 testImg[pixPos[0]][pixPos[1]] = value
            b += 1
        cv2.imshow('Result', testImg)
        cv2.waitKey(0)
        cv2.imwrite("./result/regionlist.jpg", testImg)
    """
                 