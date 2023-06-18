import cv2
import numpy as np
from matplotlib import pyplot as plt

#よっしゃ、さっそく画像読み込むで～！！！
img = cv2.imread('ここにファイル名を入力')
assert img is not None, 'Failed to load image.'

#グレースケールに変換しちゃう
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#閾値の設定せなあかんで(黒が 0 で白が 255 設定範囲は 0～255)
threshold = 230
#閾値より明るいとこは白、それより暗いとこは黒になるで
#「Contours.JPG」って名前のファイルが保存されて、輪郭が赤色で示されるから、
#その画像を見て求めたい面積と一致するように閾値を変更してみてくれ

#ほんなら二値化するわ
ret, img_binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

#二値化した画像に存在する輪郭を抽出したついでに、情報も書き出しとくで
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,0,255), 2)

#全体の画素数数えるで
whole_area=img_binary.size
#白部分の画素数数えるで
white_area=cv2.countNonZero(img_binary)
#黒部分の画素数数えるで
black_area=whole_area-white_area

#それぞれの割合を表示すんで
#画像全体の面積を定規かなんかで測っといて、割合から面積出してみてちょ
print('White_Area = '+str(white_area/whole_area*100)+'%')
print('Black_Area = '+str(black_area/whole_area*100)+'%')

#参考までに処理中の画像表示してみるわ
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
plt.title('Grayscale')
plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(img_binary, cv2.COLOR_BGR2RGB))
plt.title('Binalized')
plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Contours')

#輪郭の情報を書き込んだ画像保存しとくからよー見てみて
ret = cv2.imwrite("Contours.JPG", img)
assert ret, 'Image could not be saved.'