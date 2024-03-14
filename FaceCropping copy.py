#顔器官の切り抜き

#-*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
from imutils import face_utils

class FaceCropping:
    #顔特徴のランドマーク68点の座標を取得[68 * 2]
    def __setFaceLandmark(self):
        #学習済みモデルのパスを指定
        detector = dlib.get_frontal_face_detector()
        learned_model_path ="./learned-models/"
        predictor = dlib.shape_predictor(learned_model_path + 'shape_predictor_68_face_landmarks.dat')   
        
        gray = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)#処理を早くするためグレースケールに変換  
        faces = detector(gray, 1)
        for face in faces:
            landmark = predictor(gray, face)
            landmark = face_utils.shape_to_np(landmark)
            self.__landmark = landmark
            return
    
    #ランドマークから顔器官の(top, bottom, left, right)[4 * 2]を取得
    #各顔器官の上下左右の端に位置するランドマーク
    #右眉
    def getRightEyebrowLandmark(self):
        rightEyebrowTop = self.__landmark[19]
        rightEyebrowBottom = self.__landmark[17]
        if self.__landmark[17][1] > self.__landmark[21][1]:
            rightEyebrowBottom = self.__landmark[17]
        rightEyebrowLeft = self.__landmark[17]
        rightEyebrowRight = self.__landmark[21]

        return [rightEyebrowTop, rightEyebrowBottom, rightEyebrowLeft, rightEyebrowRight]
    #右目
    def getRightEyeLandmark(self):
        rightEyeTop = self.__landmark[38]
        if self.__landmark[37][1] < self.__landmark[38][1]:
            rightEyeTop = self.__landmark[37]
        rightEyeBottom = self.__landmark[41]
        if self.__landmark[40][1] > self.__landmark[41][1]:
            rightEyeBottom = self.__landmark[40]
        rightEyeLeft = self.__landmark[36]
        rightEyeRight = self.__landmark[39]
    
        return [rightEyeTop, rightEyeBottom, rightEyeLeft, rightEyeRight]
    #左眉
    def getLeftEyebrowLandmark(self):
        leftEyebrowTop = self.__landmark[24]
        leftEyebrowBottom = self.__landmark[26]
        if self.__landmark[22][1] > self.__landmark[26][1]:
            leftEyebrowBottom = self.__landmark[22]
        leftEyebrowLeft = self.__landmark[22]
        leftEyebrowRight = self.__landmark[26]

        return [leftEyebrowTop, leftEyebrowBottom, leftEyebrowLeft, leftEyebrowRight]
    #左目
    def getLeftEyeLandmark(self):
        leftEyeTop = self.__landmark[44]
        if self.__landmark[43][1] < self.__landmark[44][1]:
            leftEyeTop = self.__landmark[43]    
        leftEyeBottom = self.__landmark[47]
        if self.__landmark[46][1] < self.__landmark[47][1]:
            leftEyeBottom = self.__landmark[46]
        leftEyeLeft = self.__landmark[42]
        leftEyeRight = self.__landmark[45]

        return [leftEyeTop, leftEyeBottom, leftEyeLeft, leftEyeRight]
    #口
    def getMouseLandmark(self):
        tempList = np.array(self.__landmark[50])
        tempList = np.block([[tempList], [self.__landmark[51]]])
        tempList = np.block([[tempList], [self.__landmark[52]]])
        tempIndex = np.argmin(tempList, axis=0)
        mouseTop = tempList[tempIndex[1], :]

        tempList = np.array(self.__landmark[56])
        tempList = np.block([[tempList], [self.__landmark[57]]])
        tempList = np.block([[tempList], [self.__landmark[58]]])
        tempIndex = np.argmax(tempList, axis=0)
        mouseBottom = tempList[tempIndex[1], :]
    
        mouseLeft = self.__landmark[48]
        mouseRight = self.__landmark[54]

        return [mouseTop, mouseBottom, mouseLeft, mouseRight]
    #鼻
    def getNoseLandmark(self):
        noseTop = self.__landmark[27]
        noseBottom = self.__landmark[33]
        noseLeft = self.__landmark[31]
        noseRight = self.__landmark[35]
        
        return [noseTop, noseBottom, noseLeft, noseRight]

    #EyebrowBottom < EyeTop なら,距離の1/2
    #逆の場合はどうしよう?

    #顔器官の写る領域をトリミングする際に必要なtop,bottom,left,rightのリスト[5 * 4 * 1]を返す
    def __setFaceOrganCoordList(self):
        rightEyebrow = self.getRightEyebrowLandmark()
        rightEye = self.getRightEyeLandmark()
        leftEyebrow = self.getLeftEyebrowLandmark()
        leftEye = self.getLeftEyeLandmark()
        mouse = self.getMouseLandmark()
        nose = self.getNoseLandmark()
                
        #それぞれの顔器官の一回り外側から切り抜けるように調節
        #右眉トリミング領域
        cropRightEyebrowTop = round(max(0, rightEyebrow[0][1] - (rightEye[0][1] - (rightEyebrow[1][1] + self.__landmark[21][1]) / 2) / 2))
        #cropRightEyebrowBottom = round(rightEyebrow[1][1] + (rightEye[0][1] - (rightEyebrow[1][1] + self.__landmark[21][1]) / 2) / 2)
        cropRightEyebrowBottom = round(min(rightEyebrow[1][1], rightEye[0][1]))
        cropRightEyebrowLeft = round(rightEyebrow[2][0] - (self.__landmark[22][0] - self.__landmark[21][0]) / 2)
        cropRightEyebrowRight = round(rightEyebrow[3][0] + (self.__landmark[22][0] - self.__landmark[21][0]) / 2)
        rightEyebrowCoordList = [cropRightEyebrowTop, cropRightEyebrowBottom, cropRightEyebrowLeft, cropRightEyebrowRight]
        #右目トリミング領域
        cropRightEyeTop = round(rightEye[0][1] - (rightEye[0][1] - (rightEyebrow[1][1] + self.__landmark[21][1]) / 2) / 2)
        cropRightEyeBottom = round(rightEye[1][1] + (rightEye[0][1] - (rightEyebrow[1][1] + self.__landmark[21][1]) / 2) / 2) + 10
        cropRightEyeLeft = round(rightEye[2][0] - (self.__landmark[27][0] - self.__landmark[39][0]) / 2)
        cropRightEyeRight = round(rightEye[3][0] + (self.__landmark[27][0] - self.__landmark[39][0]) / 2)
        rightEyeCoordList = [cropRightEyeTop, cropRightEyeBottom, cropRightEyeLeft, cropRightEyeRight]
        #左眉トリミング領域
        cropLeftEyebrowTop = round(max(0, leftEyebrow[0][1] - (leftEye[0][1] - (leftEyebrow[1][1] + self.__landmark[22][1]) / 2) / 2))
        #cropLeftEyebrowBottom = round(leftEyebrow[1][1] + (leftEye[0][1] - (rightEyebrow[1][1] + self.__landmark[22][1]) / 2) / 2)
        cropLeftEyebrowBottom = round(min(leftEyebrow[1][1], leftEye[0][1]))
        cropLeftEyebrowLeft = round(leftEyebrow[2][0] - (self.__landmark[22][0] - self.__landmark[21][0]) / 2)
        cropLeftEyebrowRight = round(leftEyebrow[3][0] + (self.__landmark[22][0] - self.__landmark[21][0]) / 2)
        leftEyebrowCoordList = [cropLeftEyebrowTop, cropLeftEyebrowBottom, cropLeftEyebrowLeft, cropLeftEyebrowRight]
        #左目トリミング領域
        cropLeftEyeTop = round(leftEye[0][1] - (leftEye[0][1] - (rightEyebrow[1][1] + self.__landmark[22][1]) / 2) / 2)
        cropLeftEyeBottom = round(leftEye[1][1] + (leftEye[0][1] - (rightEyebrow[1][1] + self.__landmark[22][1]) / 2) / 2) + 10
        cropLeftEyeLeft = round(leftEye[2][0] - (self.__landmark[42][0] - self.__landmark[27][0]) / 2)
        cropLeftEyeRight = round(leftEye[3][0] + (self.__landmark[42][0] - self.__landmark[27][0]) / 2)
        leftEyeCoordList = [cropLeftEyeTop, cropLeftEyeBottom, cropLeftEyeLeft, cropLeftEyeRight]
        #口トリミング領域
        cropMouseTop = round(mouse[0][1] - (mouse[0][1] - self.__landmark[33][1]) / 2)
        cropMouseBottom = round(mouse[1][1] + (mouse[0][1] - self.__landmark[33][1]) / 2)
        cropMouseLeft = round(mouse[2][0] - (mouse[0][1] - self.__landmark[33][1]) / 2)
        cropMouseRight = round(mouse[3][0] + (mouse[0][1] - self.__landmark[33][1]) / 2)
        mouseCoordList = [cropMouseTop, cropMouseBottom, cropMouseLeft, cropMouseRight]
        #鼻トリミング領域
        #noseCoordList = [round(nose[0][1]), round(nose[1][1]), round((nose[2][0] + self.__landmark[39][0]) / 2), round((nose[3][0] + self.__landmark[42][0]) / 2)]
        noseCoordList = [round(nose[0][1]), round(nose[1][1] + 10), round(self.__landmark[31][0] - 10), round(self.__landmark[35][0] + 10)]

        self.__faceOrganCoordList = [rightEyebrowCoordList, rightEyeCoordList, leftEyebrowCoordList, leftEyeCoordList, mouseCoordList, noseCoordList]
        
    #顔器官が写る領域を切り抜いた画像を取得
    def __setCroppedImgList(self):
        self.__croppedImgList = []
        self.__setFaceOrganCoordList()
        for faceOrganCoord in self.__faceOrganCoordList:
            #元画像からtop, bottom, left, rightの4つの値を指定して、顔器官をそれぞれ切り抜く
            organImg = self.__img.copy()
            organImg = organImg[faceOrganCoord[0]:faceOrganCoord[1], faceOrganCoord[2]:faceOrganCoord[3]]
            self.__croppedImgList.append(organImg)

    def __init__(self, imgPath):
        self.__img = cv2.imread(imgPath)
        self.__setFaceLandmark()
        self.__setCroppedImgList()

    def draw_landmark(self):
        for points in self.__landmark:
            cv2.drawMarker(self.__img, (points[0], points[1]), (0, 255, 0), markerSize=8, thickness=2)
        return self.__img

    def getFaceLandmark(self):
        return self.__landmark
    def getFaceOrganCoordList(self):
        return self.__faceOrganCoordList
    def getCroppedImgList(self):
        return self.__croppedImgList

    def showImgList(self):
        cv2.imshow("img_eyebrow_right", self.__croppedImgList[0])
        cv2.moveWindow('img_eyebrow_right', 100, 100)
        cv2.imshow("img_eye_right", self.__croppedImgList[1])
        cv2.moveWindow('img_eye_right', 500, 100)
        cv2.imshow("img_eyebrow_left", self.__croppedImgList[2])
        cv2.moveWindow('img_eyebrow_left', 100, 500)
        cv2.imshow("img_eye_left", self.__croppedImgList[3])
        cv2.moveWindow('img_eye_left', 500, 500)
        cv2.imshow("img_mouse", self.__croppedImgList[4])
        cv2.moveWindow('img_mouse', 1000, 500)
        cv2.imshow("img_nose", self.__croppedImgList[5])
        cv2.moveWindow('img_nose', 1200, 700)

        cv2.imshow("input_img", self.__img)
        cv2.moveWindow('input_img', 1400, 900)
        cv2.waitKey(0)
    
    def saveImgList(self):
        cv2.imwrite("./result/faceCropping/eyebrow_right.jpg", self.__croppedImgList[0])
        cv2.imwrite("./result/faceCropping/eye_right.jpg", self.__croppedImgList[1])
        cv2.imwrite("./result/faceCropping/eyebrow_left.jpg", self.__croppedImgList[2])
        cv2.imwrite("./result/faceCropping/eye_left.jpg", self.__croppedImgList[3])
        cv2.imwrite("./result/faceCropping/mouse.jpg", self.__croppedImgList[4])
        cv2.imwrite("./result/faceCropping/nose.jpg", self.__croppedImgList[5])
        cv2.imwrite("./result/faceCropping/input_marked.jpg", self.__img)
    
    def save_content_img_list(self):
        cv2.imwrite("./result/faceCropping/content/eyebrow_right.jpg", self.__croppedImgList[0])
        cv2.imwrite("./result/faceCropping/content/eye_right.jpg", self.__croppedImgList[1])
        cv2.imwrite("./result/faceCropping/content/eyebrow_left.jpg", self.__croppedImgList[2])
        cv2.imwrite("./result/faceCropping/content/eye_left.jpg", self.__croppedImgList[3])
        cv2.imwrite("./result/faceCropping/content/mouse.jpg", self.__croppedImgList[4])
        cv2.imwrite("./result/faceCropping/content/nose.jpg", self.__croppedImgList[5])

    def save_style_img_list(self):
        cv2.imwrite("./result/faceCropping/style/eyebrow_right.jpg", self.__croppedImgList[0])
        cv2.imwrite("./result/faceCropping/style/eye_right.jpg", self.__croppedImgList[1])
        cv2.imwrite("./result/faceCropping/style/eyebrow_left.jpg", self.__croppedImgList[2])
        cv2.imwrite("./result/faceCropping/style/eye_left.jpg", self.__croppedImgList[3])
        cv2.imwrite("./result/faceCropping/style/mouse.jpg", self.__croppedImgList[4])
        cv2.imwrite("./result/faceCropping/style/nose.jpg", self.__croppedImgList[5])

    # マスク画像の作成
    ## points内の領域を 255 で埋める
    def draw_convex_hull(self, img, points):
        points = cv2.convexHull(points)
        #print(points)
        cv2.fillConvexPoly(img, points, color = (255, 255, 255))
        return img

    def get_eyebrow_right_mask_img(self):
        img = np.zeros(self.__croppedImgList[0].shape[:2], dtype = np.float64)
        new_start_point = [self.__faceOrganCoordList[0][2], self.__faceOrganCoordList[0][0]]
        points = np.array([self.__landmark[17] - new_start_point + [-11, 11], 
                           self.__landmark[18] - new_start_point + [-11, -5], 
                           self.__landmark[19] - new_start_point + [0, -5], 
                           self.__landmark[20] - new_start_point + [11, -5], 
                           self.__landmark[21] - new_start_point + [11, 11]])
        img = self.draw_convex_hull(img, points)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    
    def get_eye_right_mask_img(self):
        img = np.zeros(self.__croppedImgList[1].shape[:2], dtype = np.float64)
        new_start_point = [self.__faceOrganCoordList[1][2], self.__faceOrganCoordList[1][0]]
        points = np.array([self.__landmark[36] - new_start_point + [-16, 0], 
                           self.__landmark[37] - new_start_point + [-11, -11], 
                           self.__landmark[38] - new_start_point + [11, -11], 
                           self.__landmark[39] - new_start_point + [11, 0], 
                           self.__landmark[40] - new_start_point + [11, 11], 
                           self.__landmark[41] - new_start_point + [-11, 11]])
        img = self.draw_convex_hull(img, points)
        img = cv2.blur(img, (7, 7))
        img = cv2.blur(img, (7, 7))
        img = cv2.blur(img, (7, 7))
        
        return img
    
    def get_eyebrow_left_mask_img(self):
        img = np.zeros(self.__croppedImgList[2].shape[:2], dtype = np.float64)
        new_start_point = [self.__faceOrganCoordList[2][2], self.__faceOrganCoordList[2][0]]
        points = np.array([self.__landmark[22] - new_start_point + [-11, 11], 
                           self.__landmark[23] - new_start_point + [-11, -5], 
                           self.__landmark[24] - new_start_point + [0, -5], 
                           self.__landmark[25] - new_start_point + [11, -5], 
                           self.__landmark[26] - new_start_point + [11, 11]])
        img = self.draw_convex_hull(img, points)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def get_eye_left_mask_img(self):
        img = np.zeros(self.__croppedImgList[3].shape[:2], dtype = np.float64)
        new_start_point = [self.__faceOrganCoordList[3][2], self.__faceOrganCoordList[3][0]]
        points = np.array([self.__landmark[42] - new_start_point + [-11, 0], 
                           self.__landmark[43] - new_start_point + [-11, -11], 
                           self.__landmark[44] - new_start_point + [11, -11], 
                           self.__landmark[45] - new_start_point + [16, 0], 
                           self.__landmark[46] - new_start_point + [11, 11], 
                           self.__landmark[47] - new_start_point + [-11, 11]])
        img = self.draw_convex_hull(img, points)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        img = cv2.GaussianBlur(img, (7, 7), 0)
        
        return img
    
    def get_mouse_mask_img(self):
        img = np.zeros(self.__croppedImgList[4].shape[:2], dtype = np.float64)
        new_start_point = [self.__faceOrganCoordList[4][2], self.__faceOrganCoordList[4][0]]
        points = np.array([self.__landmark[48] - new_start_point, 
                           self.__landmark[49] - new_start_point, 
                           self.__landmark[50] - new_start_point, 
                           self.__landmark[51] - new_start_point, 
                           self.__landmark[52] - new_start_point, 
                           self.__landmark[53] - new_start_point, 
                           self.__landmark[54] - new_start_point, 
                           self.__landmark[55] - new_start_point, 
                           self.__landmark[56] - new_start_point, 
                           self.__landmark[57] - new_start_point, 
                           self.__landmark[58] - new_start_point, 
                           self.__landmark[59] - new_start_point])
        img = self.draw_convex_hull(img, points)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def get_nose_mask_img(self):
        img = np.zeros(self.__croppedImgList[5].shape[:2], dtype = np.float64)
        new_start_point = [self.__faceOrganCoordList[5][2], self.__faceOrganCoordList[5][0]]
        points = np.array([self.__landmark[27] - new_start_point, 
                           #self.__landmark[28] - new_start_point, 
                           #self.__landmark[29] - new_start_point, 
                           #self.__landmark[30] - new_start_point, 
                           [self.__landmark[35][0] - new_start_point[0], round((self.__landmark[27][1] + self.__landmark[33][1]) / 2) - new_start_point[1]],
                           self.__landmark[35] - new_start_point + [11, 0],
                           self.__landmark[34] - new_start_point, 
                           self.__landmark[33] - new_start_point, 
                           self.__landmark[32] - new_start_point, 
                           self.__landmark[31] - new_start_point + [-11, 0], 
                           [self.__landmark[31][0] - new_start_point[0], round((self.__landmark[27][1] + self.__landmark[33][1]) / 2) - new_start_point[1]],
                           ])
        img = self.draw_convex_hull(img, points)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    
    def get_mask_img_list(self):
        mask_img_list = []
        
        mask_img_list.append(self.get_eyebrow_right_mask_img())
        mask_img_list.append(self.get_eye_right_mask_img())
        mask_img_list.append(self.get_eyebrow_left_mask_img())
        mask_img_list.append(self.get_eye_left_mask_img())
        mask_img_list.append(self.get_mouse_mask_img())
        mask_img_list.append(self.get_nose_mask_img())

        return mask_img_list
    
"""
#test = FaceCropping("./content/test/bustshot.jpg")
test = FaceCropping("./style/Jeanne/Jeanne_Samary.jpg")

test.draw_landmark()

test.showImgList()
test.saveImgList()
"""

"""
test = FaceCropping("./content/test/bustshot.jpg")
img = test.getCroppedImgList()[5]
img_copy = img.copy()
#img_copy[True] = [255, 0, 0]

#mask_img = test.get_eyebrow_right_mask_img()
mask_img = test.get_nose_mask_img()
 
#img[mask_img != 255] = [128, 128, 128]
#img = cv2.addWeighted(img, mask_img, img_copy, 1 - mask_img, 0)

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        img_copy[x, y] = img[x, y] * (mask_img[x, y] / [255, 255, 255])

cv2.imshow("img", img) 
cv2.imshow("mask", mask_img)
cv2.imshow("result", img_copy) 
cv2.imwrite("result_test.jpg", img_copy) 
cv2.waitKey()
"""

