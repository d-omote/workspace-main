# coding: utf-8
import time
start_time = time.perf_counter()

#WARNING非表示用
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import cv2
import numpy as np

import composit2paintingMask as c2pMask

#入力画像のパス
content_image_path = './content/test/bustshot.jpg'
#content_image_path = './content/test/model_1.jpg'

style_image_path  = './style/Jeanne/Jeanne_Samary.jpg'
nope_image_path = './style/Jeanne/Jeanne_Samary_nope.jpg'

#style_image_path  = './style/Femme/Femme.jpg'
#nope_image_path = './style/Femme/Femme_nope.jpg'

#合成の準備
c2p_test = c2pMask.Composit2paintingMask(content_image_path, style_image_path, nope_image_path)

#合成
img = c2p_test.get_masked_composited_img()
end_time = time.perf_counter()
print("総処理時間：" + str(end_time - start_time))

#結果の出力
cv2.imshow('sample', np.array(img, dtype='uint8'))
cv2.waitKey()
cv2.imwrite('test.jpg', img)

"""
#ある画素から領域拡張はできる(筆跡を1つだけ作成
region = RegionGrow.RegionColorGrow.fourNeighbor(styleImageAdress, 450, 390, 150)
region.show()
region.save()
"""

"""
#画像全体を領域分割
regionList = RCL.RegionColorList(test_image_path)
regionList.setRegionList()
regionList.showRegionImg()
"""

"""
tf.config.list_physical_devices('GPU')
#print(tf.test.is_gpu_available())
from tensorflow.python.client import device_lib
print()
print(device_lib.list_local_devices())
"""

"""
# スタイル変換(tensorflow)
print("aaa")
content_image_path = "./content/test/content_eye_left.jpg" 
style_image_path = "./style/Jeanne/Jeanne_eye_left.jpg"
print("bbb")
image = st.main(content_image_path,style_image_path)
print("hhh")
image.show()
#image.save('result/styletransfer/result_bustshot2.jpg', quality=95)
"""

"""
fc = FaceCropping.FaceCropping(test_image_path)
fc.draw_landmark()
fc.showImgList()
fc.saveImgList()
"""

"""
st = styleTransfer.main('./result/faceCropping/content/eye_left.jpg', './result/faceCropping/style/eye_left.jpg')
st.show()
"""

"""
c2p_test = c2p.Composit2painting(content_image_path, style_image_path, nope_image_path)
img = c2p_test.get_compit2painting_img2()
cv2.imshow('sample', img)
cv2.imwrite('test.jpg', img)
cv2.waitKey()
"""
