from typing import Any

import matplotlib.pyplot as plt
import tensorflow as tf


def show_imgs(images, row: int, col: int) -> None:
	"""PILimagesをrow*colとして表示

	Args:
		images: PILimagesを含む
		row: plt.subplotの行
		col: plt.subplotの列
	"""
	if len(images) != (row * col):
		raise ValueError("Invalid imgs len:{} col:{} row:{}".format(len(images), row, col))

	for i, img in enumerate(images):
		plot_num = i+1
		plt.subplot(row, col, plot_num)
		plt.tick_params(labelbottom=False)		# x軸の削除
		plt.tick_params(labelleft=False)		# y軸の削除
		plt.imshow(img)
	plt.show()
	return

def show_generator_sample(generator: Any, image_path: str) -> None:
	"""ジェネレーターによって行われる前処理のサンプルを表示する

	Args:
		generator: 使用するジェネレーター
		image_path: サンプルとして表示する画像
	"""
	img = tf.keras.utils.load_img(image_path)			# 画像ファイルをPIL形式でオープン
	x =  tf.keras.utils.img_to_array(img)				# PIL形式をnumpyのndarray形式に変換
	x = x.reshape((1,) + x.shape)						# (height, width, 3) -> (1, height, width, 3)

	max_img_num = 16
	imgs = []
	for d in generator.flow(x, batch_size=1):
		imgs.append(tf.keras.utils.array_to_img(d[0], scale=True))			# このあと画像を表示するためにndarrayをPIL形式に変換して保存する
		if (len(imgs) % max_img_num) == 0:									# datagen.flowは無限ループするため必要な枚数取得できたらループを抜ける
			break
	show_imgs(imgs, row=4, col=4)
	return
