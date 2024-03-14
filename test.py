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
