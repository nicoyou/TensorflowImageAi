import glob
import pathlib
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EXTENSION_LIST = [
	"png",
	"jpg",
	"jpeg",
	"jfif",
	"webp",
]

def load_image(image_path: str) -> Any:
	"""画像をopenCVで読み込む ( ファイルパスに日本語を含む場合でも動作する )

	Args:
		image_path: 画像のファイルパス

	Returns:
		openCV形式の画像オブジェクト
	"""
	pil_img = Image.open(image_path)					# Pillowで画像ファイルを開く
	image = np.array(pil_img)							# PillowからNumPyへ変換
	pil_img = pil_img.convert("RGB")					# カラー画像に変換する
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)		# RGBからBGRへ変換する
	return image

def image_comparison(images: tuple | list) -> None:
	"""２枚の画像を同時に描画して比較する

	Args:
		images: openCV形式の画像を 2 つ格納したリスト
	"""
	fig = plt.figure(dpi=160)
	for i, im in enumerate(images):
		fig.add_subplot(1, 2, i + 1).set_title(str(i))
		plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

	# plt.get_current_fig_manager().full_screen_toggle()		# フルスクリーン
	plt.show()
	return

def get_image_path_from_dir(dir_path: str) -> None:
	"""ディレクトリ内全ての画像を再帰的に検索する

	Args:
		dir_path: 画像を検索するディレクトリ

	Returns:
		検出した画像パスの一覧
	"""
	images = glob.glob(str(pathlib.Path(dir_path, "**")), recursive=True)
	image_path_list = []
	for image in images:
		image_path = pathlib.Path(image)
		if image_path.suffix[1:].lower() in EXTENSION_LIST and image_path.is_file():
			image_path_list.append(image_path)
	return image_path_list
