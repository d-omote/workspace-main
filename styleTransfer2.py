# coding: utf-8
import time

#遅い
a = time.perf_counter()
import tensorflow as tf
print("tensorflowのインポート完了(所要時間：" + str(time.perf_counter() - a) + "秒)")
print()

import tensorflow_hub as hub
import numpy as np
import PIL.Image

#img_path -> 'tensorflow.python.framework.ops.EagerTensor'
def load_img(path_to_img):
  #最大寸法を 512 ピクセルに制限
  max_dim = 512
  
  #これが問題あり
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)

  img = tf.image.convert_image_dtype(img, tf.float32)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  img = tf.image.resize(img, new_shape)

  img = img[tf.newaxis, :]
  
  return img

"""
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
"""

#'tensorflow.python.framework.ops.EagerTensor' -> PIL 
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

#PILイメージで返す
def main(content_path, style_path):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    """
    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')
    plt.pause(0.001)

    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')
    plt.pause(0.001)
    """
    hub_model = hub.load('./model')
    
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    return tensor_to_image(stylized_image)

# 切り抜いた画像リストをスタイル変換する(imgpath->PIL)
def get_styled_img_list(content_img_path_list, style_img_path_list):
  styled_img_list = []
  hub_model = hub.load('./model')

  for i in range(len(content_img_path_list)):
    content_image = load_img(content_img_path_list[i])
    style_image = load_img(style_img_path_list[i])
    
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    
    styled_img_list.append(tensor_to_image(stylized_image))

  return styled_img_list 
