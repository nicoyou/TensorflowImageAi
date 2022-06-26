import glob
import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image

import anime_face_detector

EXTENSION_LIST = [
	"png",
	"jpg",
	"jpeg",
	"jfif",
	"webp",
]
OUTPUT_DIR = "output"


# 画像を読み込む
def load_image(image_path):
	pil_img = Image.open(image_path)					# Pillowで画像ファイルを開く
	image = np.array(pil_img)							# PillowからNumPyへ変換
	pil_img = pil_img.convert("RGB")					# カラー画像に変換する
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)		# RGBからBGRへ変換する
	return image

# ２枚の画像を同時に描画して比較する
def image_comparison(images):
	fig = plt.figure(dpi=160)
	for i, im in enumerate(images):
		fig.add_subplot(1, 2, i + 1).set_title(str(i))
		plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

	# plt.get_current_fig_manager().full_screen_toggle()		# フルスクリーン
	plt.show()
	return

# ディレクトリ内の全ての画像を取得する
def get_image_path_from_dir(dir_path):
	images = glob.glob(str(pathlib.Path(dir_path, "**")), recursive=True)
	image_path_list = []
	for image in images:
		image_path = pathlib.Path(image)
		if image_path.suffix[1:].lower() in EXTENSION_LIST and image_path.is_file():
			image_path_list.append(image_path)
	return image_path_list

# キャラクターの画像から顔を切り抜く
def crop_image_face(overwrite = False):
	afd = anime_face_detector.AnimeFaceDetector()
	images = get_image_path_from_dir("./dataset")
	for image in tqdm.tqdm(images):
		out_path = pathlib.Path(OUTPUT_DIR, image)
		if overwrite or not (out_path.parent / (out_path.stem + f"_{0:02}.png")).is_file():
			os.makedirs(out_path.parent, exist_ok=True)
			faces = afd.get_faces(image)

			if len(faces) >= 1:
				pil_image = Image.open(image)
				for i, face in enumerate(faces):
					im_crop = pil_image.crop(face["bbox"])
					im_crop.save(out_path.parent / (out_path.stem + f"_{i:02}.png"))

if __name__ == "__main__":
	crop_image_face()
