import glob

import cv2
import numpy as np
import tqdm
from PIL import Image


# 検出した顔の数を取得する
def check_face_image(image_path, im_show = False):
	cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
	try:
		pil_img = Image.open(image_path)			# Pillowで画像ファイルを開く
		image = np.array(pil_img)					# PillowからNumPyへ変換
		# カラー画像のときは、RGBからBGRへ変換する
		if image.ndim == 3:
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		gray = cv2.equalizeHist(gray)
	except Exception as e:
		print(e)
		return 0

	faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))

	if len(faces) and im_show:
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)				# 検出結果を描画
		cv2.imshow("result", image)
		cv2.waitKey(0)
	return len(faces)



if __name__ == "__main__":
	images = glob.glob("image/load_image/*")
	true_count = 0
	false_count = 0
	for row in tqdm.tqdm(images):
		if check_face_image(row):
			true_count += 1
		else:
			false_count += 1

	print(f"all: {len(images)}")
	print(f"true: {true_count}")
	print(f"false: {false_count}")
