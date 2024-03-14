import time
import cv2
import numpy as np

import FaceCropping
import styleTransfer2 as st

"""
memo
・landmark, OpenCVは(x, y)
・変換行列も(x, y)
・npは(y, x),(height, width)
"""

#人物の顔パーツを絵画に合成するクラス
class Composit2paintingMask:
    # 写真から絵画への拡大率を取得：絵画の顔の縦幅 / 写真の顔の縦幅
    def __get_rate_photo2painting(self, content_landmark, painting_landmark):
        content_top = (content_landmark[19] + content_landmark[24]) / 2
        content_bottom = content_landmark[8]
        content_height = np.sqrt((content_bottom[0] - content_top[0]) ** 2 + (content_bottom[1] - content_top[1]) ** 2)
        
        painting_top = (painting_landmark[19] + painting_landmark[24]) / 2
        painting_bottom = painting_landmark[8]
        painting_height = np.sqrt((painting_bottom[0] - painting_top[0]) ** 2 + (painting_bottom[1] - painting_top[1]) ** 2)
        
        return painting_height / content_height
    
    """
    # 切り抜いた各パーツの中心座標を取得
    def get_center_coord_list(face_coord_list):
        center_coord_list = []
        #organ_coordには画像の[top, bottom, left, right]がある
        for organ_coord in face_coord_list:
            center_coord_list.append([(organ_coord[2] + organ_coord[3]) / 2, (organ_coord[0] + organ_coord[1]) / 2])
        
        return center_coord_list
    """
    
    # 切り抜いた各パーツ画像の頂点を取得
    def get_parts_vertex_list(self, face_coord_list):
        parts_vertex_list = []
        for parts_vertex in face_coord_list:
            #左上、右上、右下、左下の頂点の座標を取得
            parts_vertex_list.append([[parts_vertex[2], parts_vertex[0]],[parts_vertex[3], parts_vertex[0]], [parts_vertex[3], parts_vertex[1]], [parts_vertex[2], parts_vertex[1]]])
        
        return parts_vertex_list
    
    def __init__(self, content_img_path, painting_img_path, nope_img_path): 
        # 合成先の絵画の画像
        self.__painting_img = cv2.imread(painting_img_path)
        self.__nope_img = cv2.imread(nope_img_path)

        #顔器官の切り抜き 
        start_crop_time = time.perf_counter()
        fc_conetnt = FaceCropping.FaceCropping(content_img_path)
        fc_style = FaceCropping.FaceCropping(painting_img_path)
        end_crop_time = time.perf_counter()
        print("顔パーツ画像の作成完了(所要時間：" + str(end_crop_time - start_crop_time) + "秒)")
        print()
    
        #合成先の絵画、写真の顔の特徴点
        self.__content_landmark = fc_conetnt.getFaceLandmark()
        self.__painting_landmark = fc_style.getFaceLandmark()
        # 写真,絵画それぞれの顔のパーツを切り抜いた画像リスト、
        self.__cropped_content_img_list = fc_conetnt.getCroppedImgList()
        fc_conetnt.save_content_img_list()
        fc_style.save_style_img_list()

        # スタイル変換(PILでくるのでnumpyに変換する)
        ## txtで、ファイルへのパスを管理している
        content_txt_open = open('./content_img_path.txt', 'r')
        content_path_list = content_txt_open.readlines()
        content_path_list = [line.rstrip('\n') for line in content_path_list]
        style_txt_open = open('./style_img_path.txt', 'r')
        style_path_list = style_txt_open.readlines()
        style_path_list = [line.rstrip('\n') for line in style_path_list]

        # スタイル変換
        start_styled_time = time.perf_counter()
        
        self.__styled_img_list = st.get_styled_img_list(content_path_list, style_path_list)

        end_styled_time = time.perf_counter()
        print("スタイル変換完了(所要時間：" + str(end_styled_time - start_styled_time) + "秒)")
        print()

        #PILでくるのでnumpyに変換する)
        self.__styled_img_list = [cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB) for img in self.__styled_img_list]
        
        #スタイル変換された画像を保存
        for i in range(len(self.__styled_img_list)):
            cv2.imwrite('./result/styletransfer/test'+ str(i) + '.jpg', self.__styled_img_list[i]) 
        
        
        # 切り抜いた画像をスタイル変換したリスト
        # 切り抜いた各パーツ画像の4頂点の座標のリスト
        self.__cropped_parts_vertex_list = self.get_parts_vertex_list(fc_conetnt.getFaceOrganCoordList())
        
        # 合成パーツ画像用変数の初期化
        self.__composit_img_list = []
        # スタイル変換によって画像のサイズが小さくなるので、元のコンテンツ画像に戻す
        for i in range(len(self.__styled_img_list)):
            cropped_img_h, cropped_img_w = self.__cropped_content_img_list[i].shape[:2]
            styled_img_h, styled_img_w = self.__styled_img_list[i].shape[:2]

            resized_img = cv2.resize(self.__styled_img_list[i], dsize=(cropped_img_w, int(styled_img_h * (cropped_img_w / styled_img_w))))
            self.__composit_img_list.append(resized_img)
        
        # 写真から絵画への拡大率
        self.__rate_photo2painting = self.__get_rate_photo2painting(self.__content_landmark, self.__painting_landmark)
        # 合成された絵画用の変数
        self.__composited_img = self.__nope_img.copy()

        # マスク画像の作成
        self.__mask_img_list = fc_conetnt.get_mask_img_list()

        # パーツ画像を一旦アフィン変換して合成する画像の、初期化
        self.__parts_and_back_img_list = []
        for i in range(len(self.__composit_img_list)):
            self.__parts_and_back_img_list.append(np.zeros(self.__painting_img.shape[:2], dtype = np.float64))

        # マスク画像を一旦アフィン変換して合成する画像の、初期化
        self.__mask_and_back_img_list = []
        for i in range(len(self.__composit_img_list)):
            self.__mask_and_back_img_list.append(np.zeros(self.__painting_img.shape[:2], dtype = np.float64))

    #描画パーツの大きさの設定「パーツの大きさを絵画に合うように変更」
    def get_scaling_matrix(self):
        return np.array([[self.__rate_photo2painting, 0, 0],
                         [0, self.__rate_photo2painting, 0],
                         [0, 0, 1]], np.float32)
    
    # 描画パーツの位置の設定「パーツの位置を絵画の鼻の先[30]を基準に変更」
    # 写真の鼻の先からパーツまでの距離を絵画基準に変更
    # 鼻だけは、絵画の鼻の頭の位置に合わせるだけにする
    # parts_center_coord:指定したパーツの中心座標(x, y), parts_scaled_size:指定したパーツ画像の拡張後のサイズ(width, height)
    def get_trans_matrix(self, parts_center_coord, parts_scaled_size):
        content_vec_nose2parts = parts_center_coord - self.__content_landmark[30]
        trans_x = (self.__painting_landmark[30][0] + content_vec_nose2parts[0] * self.__rate_photo2painting) - parts_scaled_size[0] / 2
        trans_y = (self.__painting_landmark[30][1] + content_vec_nose2parts[1] * self.__rate_photo2painting) - parts_scaled_size[1] / 2
        return np.array([[1, 0, trans_x],
                         [0, 1, trans_y],
                         [0, 0, 1]], np.float32)
    
    # 描画パーツの向きの決定「パーツの向きを絵画内の顔の向きに合わせる」
    def get_rotation_matrix(self):
        # 実写、絵画それぞれの顔の向き(度数法)を鼻の特徴点の座標から計算
        # 0度原点中心?向きの差を直接求める?_hu
        slope_painting_face = self.__content_landmark[27] - self.__content_landmark[30]
        slope_content_face = self.__painting_landmark[27] - self.__painting_landmark[30] 
        
        #写真内の顔の傾きから絵画内の顔の傾きの相対的な回転角度
        angle_content2angle = np.rad2deg(np.arctan2(slope_painting_face[1], slope_painting_face[0]) - 
                                         np.arctan2(slope_content_face[1], slope_content_face[0]))
        center_coord = (float(self.__painting_landmark[30][0]), float(self.__painting_landmark[30][1]))
        return np.insert(cv2.getRotationMatrix2D(center_coord, angle_content2angle, 1),
                          2, [0, 0, 1], axis=0)
       
    def get_masked_composited_img(self):
        # 合成先となる絵画のサイズ
        paint_img_h, paint_img_w = self.__painting_img.shape[:2]
        # 拡大行列の取得
        scaling_m = self.get_scaling_matrix()
        # 回転行列の取得        
        rotation_m = self.get_rotation_matrix()

        # 順番に合成
        for i in range(len(self.__composit_img_list)):
            # 各パーツ画像の中心座標
            parts_center_x = (self.__cropped_parts_vertex_list[i][0][0] + self.__cropped_parts_vertex_list[i][2][0]) / 2
            parts_center_y = (self.__cropped_parts_vertex_list[i][0][1] + self.__cropped_parts_vertex_list[i][2][1]) / 2
            # 各パーツ画像のサイズ(h,w)
            parts_img_h, parts_img_w = self.__composit_img_list[i].shape[:2]
            # 平行移動行列
            trans_m = self.get_trans_matrix([parts_center_x, parts_center_y], [parts_img_w * self.__rate_photo2painting, parts_img_h * self.__rate_photo2painting])
            
            #拡大・縮小して、平行移動して、絵画の鼻を中心に回転([3, 3]の行列の積)
            M = np.dot(rotation_m, np.dot(trans_m, scaling_m))
            self.__parts_and_back_img_list[i] = cv2.warpAffine(self.__composit_img_list[i], M[:2], (paint_img_w, paint_img_h), self.__parts_and_back_img_list[i], borderMode=cv2.BORDER_TRANSPARENT)
            self.__mask_and_back_img_list[i] = cv2.warpAffine(self.__mask_img_list[i], M[:2], (paint_img_w, paint_img_h), self.__mask_and_back_img_list[i], borderMode=cv2.BORDER_TRANSPARENT)

        for i in range(len(self.__composit_img_list)):
            cv2.imwrite("./result/Affine/parts_and_back" + str(i) + ".jpg", self.__parts_and_back_img_list[i])    
            cv2.imwrite("./result/Affine/mask_and_back" + str(i) + ".jpg", self.__mask_and_back_img_list[i])

        """
        cv2.imwrite("./2parts_and_back.jpg", self.__parts_and_back_img_list[2])
        cv2.imwrite("./2mask_and_back.jpg", self.__mask_and_back_img_list[2])
        cv2.imwrite("./3parts_and_back.jpg", self.__parts_and_back_img_list[3])
        cv2.imwrite("./3mask_and_back.jpg", self.__mask_and_back_img_list[3])        
        """

        white = np.zeros((self.__composited_img.shape))
        white += 1
        
        for i in range(len(self.__parts_and_back_img_list)):
            mask = self.__mask_and_back_img_list[i]
            mask_3 = np.stack((mask,) * 3, -1) / 255

            a = self.__parts_and_back_img_list[i] * mask_3
            b = self.__composited_img * (white - mask_3)
            self.__composited_img = a + b

        return self.__composited_img

        """
        for i in range(len(self.__parts_and_back_img_list)):
            for x in range(self.__parts_and_back_img_list[i].shape[0]):
                for y in range(self.__parts_and_back_img_list[i].shape[1]):
                    mask_pix = self.__mask_and_back_img_list[i][x, y]
                    #ここto,高速化(何回も生成している変数を先に宣言しておく上のとか?)
                    a = self.__parts_and_back_img_list[i][x, y] * (mask_pix / [255, 255, 255])
                    #b = self.__composited_img[x, y] * (([255, 255, 255] - mask_pix) / [255, 255, 255])
                    b = self.__composited_img[x, y] * (([255, 255, 255] - mask_pix) / [255, 255, 255])
                    self.__composited_img[x, y] = a + b
        """

#　ここから↓はいらない
"""
    # パーツ合成の実行関数
    def get_compit2painting_img(self):
        # 合成先となる絵画のサイズ
        paint_img_h, paint_img_w = self.__painting_img.shape[:2]
        # 拡大行列の取得
        scaling_m = self.get_scaling_matrix()
        # 回転行列の取得        
        rotation_m = self.get_rotation_matrix()

        # 順番に合成
        for i in range(len(self.__composit_img_list)):
            #各パーツ画像の中心座標
            parts_center_x = (self.__cropped_parts_vertex_list[i][0][0] + self.__cropped_parts_vertex_list[i][2][0]) / 2
            parts_center_y = (self.__cropped_parts_vertex_list[i][0][1] + self.__cropped_parts_vertex_list[i][2][1]) / 2
            #各パーツ画像のサイズ(h,w)
            parts_img_h, parts_img_w = self.__composit_img_list[i].shape[:2]
            #平行移動行列
            trans_m = self.get_trans_matrix([parts_center_x, parts_center_y], [parts_img_w * self.__rate_photo2painting, parts_img_h * self.__rate_photo2painting])
            #拡大・縮小して、平行移動して、絵画の鼻を中心に回転([3, 3]の行列の積)
            M = np.dot(rotation_m, np.dot(trans_m, scaling_m))
            self.__composited_img = cv2.warpAffine(self.__composit_img_list[i], M[:2], (paint_img_w, paint_img_h), self.__composited_img, borderMode=cv2.BORDER_TRANSPARENT)
        
        return self.__composited_img
    
    #インデックスで指定して特定のパーツだけ合成する
    def get_compit2painting_img_num(self, num):
        # 合成先となる絵画のサイズ
        paint_img_h, paint_img_w = self.__painting_img.shape[:2]
        # 拡大行列の取得
        scaling_m = self.get_scaling_matrix()
        # 回転行列の取得        
        rotation_m = self.get_rotation_matrix()

        #各パーツ画像の中心座標
        parts_center_x = (self.__cropped_parts_vertex_list[num][0][0] + self.__cropped_parts_vertex_list[num][2][0]) / 2
        parts_center_y = (self.__cropped_parts_vertex_list[num][0][1] + self.__cropped_parts_vertex_list[num][2][1]) / 2
        #各パーツ画像のサイズ(h,w)
        parts_img_h, parts_img_w = self.__composit_img_list[num].shape[:2]
        #平行移動行列
        trans_m = self.get_trans_matrix([parts_center_x, parts_center_y], [parts_img_w * self.__rate_photo2painting, parts_img_h * self.__rate_photo2painting])
        #拡大・縮小して、平行移動して、絵画の鼻を中心に回転([3, 3]の行列の積)
        M = np.dot(rotation_m, np.dot(trans_m, scaling_m))
        self.__composited_img = cv2.warpAffine(self.__composit_img_list[num], M[:2], (paint_img_w, paint_img_h), self.__composited_img, borderMode=cv2.BORDER_TRANSPARENT)
        
        return self.__composited_img
    
    #目を、鼻の上のレイヤーに合成するために、合成の順序を入れ替えた(もともとは、右繭、右目、左眉、左目、口、鼻)
    def get_compit2painting_img2(self):
        self.get_compit2painting_img_num(0)
        self.get_compit2painting_img_num(2)
        self.get_compit2painting_img_num(5)
        self.get_compit2painting_img_num(4)
        self.get_compit2painting_img_num(1)
        return self.get_compit2painting_img_num(3)

"""
