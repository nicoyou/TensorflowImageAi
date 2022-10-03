import glob
import pathlib
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_EXTENSION_LIST = [
	"png",
	"jpg",
	"jpeg",
	"jfif",
	"webp",
]

def load_image(image_path: str) -> Any:
	"""画像をopenCV ( ndarray ) で読み込む ( ファイルパスに日本語を含む場合でも動作する )

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

def image_to_pillow(image: Any) -> Any | None:
	"""numpy 形式の画像を pillow形式 に変換する

	Args:
		image: 画像オブジェクト

	Returns:
		pillow形式の画像
	"""
	if isinstance(image, Image.Image):
		return image
	if isinstance(image, np.ndarray):
		new_image = image.copy()
		if new_image.ndim == 2:				# モノクロ
			pass
		elif new_image.shape[2] == 3:		# カラー
			new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
		elif new_image.shape[2] == 4:		# 透過
			new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
		new_image = Image.fromarray(new_image)
		return new_image
	return None

def image_display(images: tuple | list) -> None:
	"""複数のの画像を同時に描画する

	Args:
		images: 画像を格納したリスト
	"""
	fig = plt.figure(dpi=160)
	for i, im in enumerate(images):
		fig.add_subplot(1, len(images), i + 1).set_title(str(i))
		plt.xticks(color="None")				# メモリの数字を消す
		plt.yticks(color="None")
		plt.tick_params(length=0)				# メモリを消す
		plt.imshow(image_to_pillow(im))

	# plt.get_current_fig_manager().full_screen_toggle()		# フルスクリーン
	plt.show()
	return

def get_image_path_from_dir(dir_path: str) -> list:
	"""ディレクトリ内全ての画像を再帰的に検索する

	Args:
		dir_path: 画像を検索するディレクトリ

	Returns:
		検出した画像パスのリスト
	"""
	images = glob.glob(str(pathlib.Path(dir_path, "**")), recursive=True)
	image_path_list = []
	for image in images:
		image_path = pathlib.Path(image)
		if image_path.suffix[1:].lower() in IMAGE_EXTENSION_LIST and image_path.is_file():
			image_path_list.append(image_path)
	return image_path_list
