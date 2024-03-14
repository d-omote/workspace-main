# coding: utf-8
import time
start_time = time.perf_counter()

#WARNING非表示用
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import cv2
import numpy as np

import composit2paintingMask as c2pMask

flag = 1
if flag == 0:
    #絵画1
    style_image_path  = './style/Jeanne/Jeanne_Samary.jpg'
    nope_image_path = './style/Jeanne/Jeanne_Samary_nope.jpg'
    filter_image_path = './style/Jeanne/Jeanne_Samary.png'
else:
    #絵画2
    style_image_path  = './style/Femme/Femme.jpg'
    nope_image_path = './style/Femme/Femme_nope.jpg'
    filter_image_path = './style/Femme/Femme.png'

#絵画の表示
art_img = cv2.imread(style_image_path)
h, w = art_img.shape[:2]
cv2.imshow('artwork' , cv2.resize(art_img , (int(w * (720 / h)), 720)))

#写真pathの初期化
content_image_path = './content/test/bustshot.jpg'
#カメラ入力の準備
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

filter_img = cv2.imread(filter_image_path, -1)
width, height =filter_img.shape[:2]
#filter_img = cv2.resize(filter_img, (int(width *  (300 / height)), 300))
filter_img = cv2.resize(filter_img, (700, 700))
width, height =filter_img.shape[:2]

#繰り返しのためのwhile文
while True:
    #カメラからの画像取得
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    texted_frame = frame.copy()
    texted_frame = cv2.resize(texted_frame, (int(1920 * (720/1080)), 720))
    #cv2.rectangle(texted_frame, (425, 50), (1280 - 425, 670), (0, 0, 255))
    texted_frame[0:height, 300:width+300] = texted_frame[0:height, 300:width+300] * (1 - filter_img[:, :, 3:] / 255) + \
                      (filter_img[:, :, :3] * (filter_img[:, :, 3:] / 255))
    cv2.putText(texted_frame, 'Press "p" to take a photograph', (10, 50),
               cv2.FONT_HERSHEY_PLAIN, 2,
               (255, 255, 255), 2, cv2.LINE_AA)

    #カメラの画像の出力
    cv2.imshow('camera' , texted_frame)

    #pキーがシャッター
    key = cv2.waitKey(20)&0xff
    if key == ord('p'):
        #フレームを保存
        cv2.imwrite('./content/test/demo_bustshot.jpg', frame)
        content_image_path = './content/test/demo_bustshot.jpg'
        break
#メモリを解放して終了するためのコマンド
cap.release()

print(frame.shape[:2])

#合成の準備
c2p_test = c2pMask.Composit2paintingMask(content_image_path, style_image_path, nope_image_path)

#合成の実行
img = c2p_test.get_masked_composited_img()
end_time = time.perf_counter()
print("総処理時間：" + str(end_time - start_time))

#結果の出力
result_img = np.array(img, dtype='uint8')
h, w = result_img.shape[:2]
cv2.putText(result_img, 'Press "q" to close', (10, 120),
               cv2.FONT_HERSHEY_PLAIN, 9,
               (255, 255, 255), 5, cv2.LINE_AA)
cv2.imshow('result', cv2.resize(result_img, (int(w * (720 / h)), 720)))
#qキーが押されるまで結果の表示を続ける
while True:
    key = cv2.waitKey()&0xff
    if key == ord('q'):
        break

#結果の保存
cv2.imwrite('./result/demo_result.jpg', img)
