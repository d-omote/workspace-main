#GPTで生成した領域拡張法

import cv2
import numpy as np

def region_growing(image, seed, threshold):
    # 画像の高さと幅を取得
    height, width = image.shape[:2]

    # 出力画像を作成し、すべてのピクセルを初期化
    segmented = np.zeros_like(image)

    # 訪問済みピクセルを追跡するための配列を作成し、すべてをFalseで初期化
    visited = np.zeros((height, width), dtype=bool)

    # シード座標をキューに追加
    queue = []
    queue.append(seed)

    # シードの値を取得
    seed_value = image[seed[0], seed[1]]

    # 領域拡張法を実行
    while len(queue) > 0:
        # 訪問予定キューから現在(調査する)ピクセルを取得
        current_pixel = queue.pop(0)
            
        # 現在のピクセルの座標
        x, y = current_pixel

        # 現在のピクセルを訪問済みにマーク
        visited[x, y] = True

        # 現在のピクセルの値
        current_value = image[x, y]

        # 出力画像に現在のピクセルの値を設定
        segmented[x, y] = current_value

        # 上下左右の近傍ピクセルをチェック
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for neighbor in neighbors:
            nx, ny = neighbor

            # 画像の範囲内かつ未訪問の場合
            if nx >= 0 and nx < height and ny >= 0 and ny < width and not visited[nx, ny]:
                neighbor_value = image[nx, ny]

                # 近傍ピクセルとシードピクセルの値が閾値以下の場合にキューに追加
                if np.abs(neighbor_value.astype(np.int64) - seed_value.astype(np.int64)) <= threshold:
                    queue.append((nx, ny))
                    visited[nx, ny] = True

    return segmented

# 入力画像のパス
image_path = './../content/now/test2.PNG'

# 画像をグレースケールで読み込む
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# シード座標
seed = (256, 256)

# 閾値
threshold = 50

# 領域拡張法を適用
result = region_growing(image, seed, threshold)

# 結果を表示
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
