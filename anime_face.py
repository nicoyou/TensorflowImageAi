import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image

import anime_face_detector


# 画像を読み込む
def load_image(image_path):
	pil_img = Image.open(image_path)			# Pillowで画像ファイルを開く
	image = np.array(pil_img)					# PillowからNumPyへ変換
	if image.ndim == 3:							# カラー画像のときは、RGBからBGRへ変換する
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image

# 検出した顔の数を取得する
def check_face_image(image_path, im_show = False):
	cascade = cv2.CascadeClassifier("lbpcascade_animeface.xml")
	try:
		image = load_image(image_path)

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
	return faces

def image_comparison(images):
	fig = plt.figure(dpi=160)
	for i, im in enumerate(images):
		fig.add_subplot(1, 2, i + 1).set_title(str(i))
		plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

	# plt.get_current_fig_manager().full_screen_toggle()
	plt.show()



if __name__ == "__main__":
	afd = anime_face_detector.AnimeFaceDetector()
	images = glob.glob("image/load_images/*")
	print(images)
	true_count = 0
	false_count = 0
	for row in tqdm.tqdm(images):
		# faces = check_face_image(row)
		# if len(faces):
		# 	true_count += 1
		# else:
		# 	false_count += 1

		# if len(ai_result[row.replace("\\", "/")]):
		# 	true_count += 1
		# else:
		# 	false_count += 1

		faces = check_face_image(row)
		img = load_image(row)
		for (x, y, w, h) in faces:
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)				# 検出結果を描画

		faces = afd.get_faces(row)
		img2 = load_image(row)
		for data in faces:
			x, y, x2, y2 = data["bbox"]
			cv2.rectangle(img2, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)

		image_comparison([img, img2])
		pass
	print(f"all: {len(images)}")
	print(f"true: {true_count}")
	print(f"false: {false_count}")


