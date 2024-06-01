import cv2
import numpy as np

# 画像を読み込む
img = cv2.imread('D:\projects\spot-the-difference-generate\data\\te3.jpg')

# グレースケールに変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二値化
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 輪郭を見つける
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 面積の閾値を設定
area_threshold = 1000

# 見つけた輪郭に基づいて元の画像をマスキングし、物体を切り抜く
for i, cnt in enumerate(contours):
    # 面積を計算
    area = cv2.contourArea(cnt)

    # 面積が閾値以上の場合のみ切り抜く
    if area > area_threshold:
        x, y, w, h = cv2.boundingRect(cnt)
        crop = img[y:y+h, x:x+w]

        # 画像をRGBAに変換
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)

        # マスクを作成
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.drawContours(mask, [cnt - [x, y]], -1, (255), thickness=cv2.FILLED)

        # マスクを適用して背景を透過させる
        masked_image = cv2.bitwise_and(crop, crop, mask=mask)
        masked_image[mask==0] = [0,0,0,0]

        # アンチエイリアシングを適用
        scale = 2.0  # 拡大率
        large = cv2.resize(masked_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)  # 拡大
        smooth_image = cv2.resize(large, (masked_image.shape[1], masked_image.shape[0]), interpolation=cv2.INTER_LINEAR)  # 縮小

        # 輪郭をぼかす
        blurred_image = cv2.GaussianBlur(smooth_image, (5, 5), 0)

        cv2.imwrite(f'object_{i}.png', blurred_image)  # 切り抜いた物体を保存

cv2.destroyAllWindows()