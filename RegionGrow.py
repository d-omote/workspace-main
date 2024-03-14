import cv2
import numpy as np
import matplotlib.pyplot as plt

from Region import RegionColor, RegionGray

#領域拡張法,筆跡の作成を行う
class RegionColorGrow:
    def bgr2hsl(pixel):
        #pixel[0]の型は8ビット符号なしのため
        pixel = [pixel[0].astype(np.int16), pixel[1].astype(np.int16), pixel[2].astype(np.int16)] 
        pixMax = max(pixel)
        pixMin = min(pixel)
        
        #色相(0 ~ 360)
        hue = 0
        if not pixMax == pixMin:
            if pixel[0] == pixMax:
                hue = 60 * ((pixel[2] - pixel[1]) / (pixMax - pixMin)) + 240
            elif pixel[1] == pixMax:
                hue = 60 * ((pixel[0] - pixel[2]) / (pixMax - pixMin)) + 120
            else:
                hue = 60 * ((pixel[1] - pixel[0]) / (pixMax - pixMin))
        #マイナスだったら360の値を足す
        if hue < 0:
            hue += 360
        
        #彩度(0 ~ 255)
        s = 0
        cnt = (pixMax + pixMin) / 2
        if cnt <= 127:
            s = (pixMax - pixMin) / (pixMax + pixMin)
        else:
            s = (pixMax - pixMin) / (510 - pixMax - pixMin)
        
        saturation = s * 255
        
        #輝度(0 ~ 255)
        lightness = (pixMax + pixMin) / 2
        
        return np.array([hue, saturation, lightness], dtype=np.uint8)
    
    def bgr2hsv(pixel):
        #pixel[0]の型は8ビット符号なしのため
        pixel = [pixel[0].astype(np.int16), pixel[1].astype(np.int16), pixel[2].astype(np.int16)] 
        pixMax = max(pixel)
        pixMin = min(pixel)
        
        #色相(0 ~ 360)
        hue = 0
        
        if not pixMax == pixMin:
            if pixel[0] == pixMax:
                hue = 60 * ((pixel[2] - pixel[1]) / (pixMax - pixMin)) + 240
            elif pixel[1] == pixMax:
                hue = 60 * ((pixel[0] - pixel[2]) / (pixMax - pixMin)) + 120
            else:
                hue = 60 * ((pixel[1] - pixel[0]) / (pixMax - pixMin))
        #マイナスだったら360の値を足す
        if hue < 0:
            hue += 360
        
        #彩度(0 ~ 255)
        saturation = ((pixMax - pixMin) / pixMax) * 255
        
        #明度(0 ~ 255)
        value = pixMax
        
        return np.array([hue, saturation, value], dtype=np.uint8)
            
    #領域拡張の条件定義
    def isRegionGrow(currentPix, neighborPix):
        #2画素間のB,G,Rの各値の差を計算
        #colorDis = abs(currentPix[0] - neighborPix[0]) + abs(currentPix[1] - neighborPix[1]) + abs(currentPix[2] - neighborPix[2])
        
        #2画素間の色相の差を計算
        #colorDis = abs(RegionColorGrow.bgr2hsl(currentPix)[0].astype(np.int16) - RegionColorGrow.bgr2hsl(neighborPix)[0].astype(np.int16))
        
        #2画素間の彩度S(HSL)の差を計算
        #colorDis = abs(RegionColorGrow.bgr2hsl(currentPix)[1].astype(np.int16) - RegionColorGrow.bgr2hsl(neighborPix)[1].astype(np.int16))
        
        #2画素間の輝度の差を計算
        #colorDis = abs(RegionColorGrow.bgr2hsl(currentPix)[2].astype(np.int16) - RegionColorGrow.bgr2hsl(neighborPix)[2].astype(np.int16))
        
        #2画素間の彩度S(HSV)の差を計算
        #colorDis = abs(RegionColorGrow.bgr2hsv(currentPix)[1].astype(np.int16) - RegionColorGrow.bgr2hsv(neighborPix)[1].astype(np.int16))

        #2画素間の明度の差を計算
        colorDis = abs(RegionColorGrow.bgr2hsv(currentPix)[2].astype(np.int16) - RegionColorGrow.bgr2hsv(neighborPix)[2].astype(np.int16))
        
        #閾値は今のところは手入力
        return colorDis < 20
        
    #4近傍で領域拡張法
    def fourNeighbor_first(imgPath, targetY, targetX):
        # 入力画像の読み込みと、高さと幅を取得
        inputImg = cv2.imread(imgPath)
        height, width = inputImg.shape[:2]
        # 領域の初期化
        region = RegionColor(height, width)
        # 訪問済みピクセル追跡用配列
        visitedList = np.zeros((height, width), dtype=bool) 
        # 訪問予定キュー[y, x]の初期化
        queue = []
        queue.append([targetY, targetX])
        
        # 領域拡張法を実行
        while len(queue) > 0:
            # 訪問予定キューから現在(調査する)ピクセルを取得
            currentPos = queue.pop(0)
            # 現在のピクセルの座標値,画素値
            currentY, currentX = currentPos
            currentPix = inputImg[targetY, targetX]
            # 訪問済み配列に現在のピクセルを登録
            visitedList[currentY, currentX] = True
            # 出力画像に現在のピクセルの値を設定(領域拡張)
            region.add(currentY, currentX, inputImg[currentY][currentX])
            
            # 上下左右の近傍ピクセルをチェック
            neighbors = [(currentY - 1, currentX), (currentY + 1, currentX), (currentY, currentX - 1), (currentY, currentX + 1)]
            for neighbor in neighbors:
                neighborY, neighborX = neighbor
                # 画像の範囲内かつ未訪問の場合(既に、領域分割された画素でないかも同時に確認している)
                if neighborY >= 0 and neighborY < height and neighborX >= 0 and neighborX < width and not visitedList[neighborY, neighborX] :
                    # 領域拡張の条件定義(隣接する2画素の色の類似度を求めて、閾値以下であるか確認)
                    neighborPix = inputImg[neighborY][neighborX].astype(np.int64)
                    # 条件に合うなら、近傍ピクセルを訪問予定キューに追加
                    if RegionColorGrow.isRegionGrow(currentPix, neighborPix):
                        queue.append((neighborY, neighborX))
                        visitedList[neighborY, neighborX] = True
        return region

    #4近傍で領域拡張法
    def fourNeighbor(imgPath, targetY, targetX, visitedList):
        # 入力画像の読み込みと、高さと幅を取得
        inputImg = cv2.imread(imgPath)
        height, width = inputImg.shape[:2]
        # 出力画像をを初期化
        region = RegionColor(height, width)
        
        # 訪問予定キュー[y, x]の初期化
        queue = []
        queue.append([targetY, targetX])
        
        # 領域拡張法を実行
        while len(queue) > 0:
            # 訪問予定キューから現在(調査する)ピクセルを取得
            currentPos = queue.pop(0)
            # 現在のピクセルの座標値,画素値
            currentY, currentX = currentPos
            currentPix = inputImg[targetY, targetX]
            # 訪問済み配列に現在のピクセルを登録
            visitedList[currentY, currentX] = True
            # 出力画像に現在のピクセルの値を設定(領域拡張)
            region.add(currentY, currentX, inputImg[currentY][currentX])
            
            # 上下左右の近傍ピクセルをチェック
            neighbors = [(currentY - 1, currentX), (currentY + 1, currentX), (currentY, currentX - 1), (currentY, currentX + 1)]
            for neighbor in neighbors:
                neighborY, neighborX = neighbor
                # 画像の範囲内かつ未訪問の場合(既に、領域分割された画素でないかも同時に確認している)
                if neighborY >= 0 and neighborY < height and neighborX >= 0 and neighborX < width and not visitedList[neighborY, neighborX] :
                    # 領域拡張の条件定義(隣接する2画素の色の類似度を求めて、閾値以下であるか確認)
                    neighborPix = inputImg[neighborY][neighborX].astype(np.int64)
                    # 条件に合うなら、近傍ピクセルを訪問予定キューに追加
                    if RegionColorGrow.isRegionGrow(currentPix, neighborPix):
                        queue.append((neighborY, neighborX))
                        visitedList[neighborY, neighborX] = True
        return region        

class RegionGrayGrow:
    def aaa(imgPath, targetY, targetX, threshold):
        # 入力画像の読み込みと、高さと幅を取得
        # 要修正？(既存のグレースケール化アルゴリズムでいいの？)
        inputImg = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        height, width = inputImg.shape[:2]
        # 出力画像をを初期化
        #outImg = np.zeros_like(inputImg)
        
        region = RegionGray(height, width)
        
        # 訪問済みピクセル追跡用配列
        visitedList = np.zeros((height, width), dtype=bool)
        # 訪問予定キュー(y, x)の初期化
        queue = []
        queue.append([targetY, targetX])
        # 初期ピクセル
        targetPix = inputImg[targetY, targetX]
        
        # 領域拡張法を実行
        while len(queue) > 0:
            # 訪問予定キューから現在(調査する)ピクセルを取得
            currentPos = queue.pop(0)
            # 現在のピクセルの座標値
            currentY, currentX = currentPos
            # 訪問済み配列に現在のピクセルを登録
            visitedList[currentY, currentX] = True
            # 出力画像に現在のピクセルの値を設定(領域拡張)
            #outImg[currentY, currentX] = inputImg[currentY, currentX]
            region.add(currentY, currentX, inputImg[currentY][currentX])
            # 上下左右の近傍ピクセルをチェック
            neighbors = [(currentY - 1, currentX), (currentY + 1, currentX), (currentY, currentX - 1), (currentY, currentX + 1)]
            for neighbor in neighbors:
                neighborY, neighborX = neighbor
                
                # 画像の範囲内かつ未訪問の場合
                if not visitedList[neighborY, neighborX] and neighborY >= 0 and neighborY < height and neighborX >= 0 and neighborX < width:
                    neighborValue = inputImg[neighborY, neighborX]
                    
                    # 領域拡張の条件定義(隣接する2画素の色の類似度を求めて、閾値以下であるか確認)
                    neighborPix = inputImg[neighborY][neighborX].astype(np.int64)
                    targetPix = inputImg[targetY][targetX].astype(np.int64)
                    colorDis = (neighborPix - targetPix) ** 2
                    #colorDis = (neighborPix[0] - targetPix[0]) ** 2 + (neighborPix[1] - targetPix[1]) ** 2 + (neighborPix[2] - targetPix[2]) ** 2
                    isCloseBright = colorDis < threshold
                    # isCloseBright = np.abs(neighborValue.astype(np.int64) - targetPix.astype(np.int64)) <= threshold                    
                    
                    # 条件に合うなら、近傍ピクセルを訪問予定キューに追加
                    if isCloseBright:
                        queue.append((neighborY, neighborX))
                        visitedList[neighborY, neighborX] = True
        return region

